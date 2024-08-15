import pandas as pd
import torch
import wandb
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models.set_models import DeepSet
from utils.data.lin_transform import cal_mcc
from utils.data.generics import freeze, unfreeze
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class TaskLinearModel(nn.Module):
    def __init__(
        self,
        observation_dim,
        latent_dim,
        num_tasks,
        amortized=False,
        sigma_s=0.1,
        log=False,
        device=None,
        sigma_obs=0.01,
    ):
        """
        Constructor for the Linear task model.

        Args:
            observation_dim: Dimension of the observed features.
            latent_dim: Dimension of the latent features.
            amortized: If True, uses a neural network to predict the causal index in an amortized fashion. While in theory one
            might then be able to use this to enable some for of causal identification at test time, in practice this is not recommended
            as convergence is much better when using free parameters for the causal index.
            sigma_s: Standard deviation of the spurious noise. If None, this is learned as an additional parameter (not recommended).
        """

        super(TaskLinearModel, self).__init__()
        self.log = log
        self.num_tasks = num_tasks
        self.A = nn.Linear(latent_dim, observation_dim, bias=False)
        self.sigma_obs = torch.tensor(
            sigma_obs
        )  # Ground truth - not generating with any noise
        if sigma_s is None:
            self.sigma_s = torch.nn.Parameter(torch.randn(1))
        else:
            self.sigma_s = torch.tensor(sigma_s)
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.amortized = amortized

        if amortized:
            self.c_index_model = DeepSet(
                dim_input=observation_dim + 1,
                num_outputs=1,
                dim_output=latent_dim,
                dim_hidden=128,
            )
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.to(self.device)

    def likelihood_loss(
        self,
        inputs,
        targets,
        transformation,
        gamma,
        causal_index=None,
        use_ground_truth=False,
        debug=False,
        c_ind_params=None,
        true_gammas=None,
    ):
        """
        Compute the MLE objective for the linear model.

        Args:
            transformation: Transformation matrix to be used for the likelihood computation.
            inputs: Input observations. Shape is (num tasks x num samples x num observed features)
            targets: Target observations. Shape is (num tasks x num samples x 1)
            causal_index: Index of the causal feature. Shape is (num tasks x num latent features)
            c_ind_params: Parameters for the causal indices. Shape is (num tasks x 1 x num latent features)
            debug: If True, returns the ground truth likelihood as well.
            use_ground_truth: If True, uses the ground truth causal index for the likelihood computation.
            gamma: Spurious variable correlation parameter. Shape is (num_tasks x num latent features x 1)

        Returns:
            torch.float: negative log-likelihood value
        """

        if causal_index is not None and causal_index.ndim == 2:
            causal_index = torch.as_tensor(causal_index).unsqueeze(1).float()

        if not use_ground_truth and self.amortized:
            c_indx = torch.sigmoid(self.c_index_model(inputs, targets))
        elif not use_ground_truth:
            c_indx = torch.sigmoid(c_ind_params)
        elif causal_index is not None and true_gammas is not None:
            c_indx = causal_index
        else:
            raise ValueError("No causal index provided but debug was set to True.")

        Mu, Sigma = self._get_distribution_params(
            causal_index=c_indx,
            inputs=inputs,
            targets=targets,
            transformation=self.A,
            gamma=(gamma if not use_ground_truth else true_gammas.unsqueeze(-1)),
        )
        log_prob = self._likelihood(Mu, Sigma, inputs)

        if debug and transformation is not None:
            with torch.no_grad():
                true_gammas = (
                    torch.ones_like(gamma)
                    if true_gammas is None
                    else true_gammas.unsqueeze(-1)
                )
                gt_mu, gt_sigma = self._get_distribution_params(
                    causal_index=causal_index,
                    inputs=inputs,
                    targets=targets,
                    transformation=transformation,
                    gamma=true_gammas,
                )
                gt_log_prob = self._likelihood(gt_mu, gt_sigma, inputs)

            return -log_prob, Mu, Sigma, -gt_log_prob, c_indx

        return -log_prob, Mu, Sigma, c_indx

    def _likelihood(self, Mu, Sigma, inputs):
        """
        Computes the log-likelihood of the inputs given the parameters of the marginal likelihood distribution.
        """
        dist = torch.distributions.MultivariateNormal(loc=Mu, covariance_matrix=Sigma)
        log_prob = dist.log_prob(inputs)

        return log_prob

    def _get_distribution_params(
        self, causal_index, inputs, targets, transformation, gamma
    ):

        """
        Computes the parameters of the marginal likelihood distribution of the inputs given the targets and task variables.
        """

        if hasattr(transformation, "weight"):  # For the linear layer
            A = transformation.weight
        else:  # For the ground truth matrix
            A = transformation

        D = torch.diag_embed(self.sigma_s**2 * (1 - causal_index) + causal_index).squeeze(1)
        z_mean = targets.unsqueeze(-1) * (gamma * (1 - causal_index.transpose(-1, -2)))
        Mu = (A @ z_mean).squeeze(-1)
        Sigma = (A @ D @ A.T) + (
            (self.sigma_obs**2) * torch.eye(self.observation_dim).to(inputs.device)
        )

        return Mu, Sigma

    def train_A(
        self,
        dataset,
        num_epochs,
        use_ground_truth=False,
        debug=False,
        batch_size=32,
        optimizer=None,
        eval_interval=None,
        fixed_gamma=None,
        warmup_epochs=0,
        use_scheduler=False,
        dtype=torch.float32,
    ):

        """
        Given observed inputs and outputs, train the linear transformation matrix A to recover the ground truth transformation
        matrix of the latent space.

        Args:
            dataset: LinearDataset object.
            num_epochs: Number of epochs to train for.
            latents: Latent variables representating the underlying factors of variation. Shape is (num tasks x num samples x num latent features).
            use_ground_truth: Flag for using the ground truth causal indices for each task.
            debug: Flag for debugging. If true, will log the loss and the ground truth loss with the true transformation matrix.
            causal_index: Causal indices for each task. Shape is (num tasks x num latent features). Required if debug is true.
            batch_size: Batch size of tasks for training.
            transformation: Ground truth transformation matrix. Shape is (num latent features x num observed features). Required if debug is true.
            warmup_epochs: Number of epochs at start of training during which the parameters for gamma are frozen. After this point, everything is trained jointly.

        Returns:

        """

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.c_params = self._init_free_params(
            size=(self.num_tasks, self.observation_dim), fixed_value=None
        ).to(device=self.device, dtype=dtype)
        self.gamma_params = self._init_free_params(
            size=(self.num_tasks, self.observation_dim),
            fixed_value=fixed_gamma,
            warmup=(warmup_epochs > 0),
        ).to(device=self.device, dtype=dtype)

        if optimizer is None:
            lr = 0.001 if use_scheduler else 5e-3
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(
                self.parameters()
            )  # set any extra parameters as a partial function object (functools)

        if use_scheduler:
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                "min",
                patience=5,
                factor=0.1,
                verbose=True,
                min_lr=1e-5,
                threshold=1e-4,
            )

        wandb.watch(self)
        train_results = []
        for epoch in range(num_epochs):
            if 0 < warmup_epochs == epoch:
                unfreeze(self.gamma_params)
            elif warmup_epochs > 0 and epoch < warmup_epochs:
                freeze(self.gamma_params)

            # Eval progress every eval_interval epochs
            # if eval_interval is not None and (epoch % eval_interval) == 0:
            #     results = self.eval_A(
            #         observations=dataset.o_supportx,
            #         latents=dataset.latents,
            #         transformation=dataset.transformation,
            #         epoch=epoch,
            #         causal_index=dataset.causal_index,
            #         true_gammas=dataset.gamma_coeffs,
            #         approx_gammas=self.gamma_params,
            #         use_ground_truth=use_ground_truth,
            #     )
            #     results['epoch'] = epoch
            #     train_results.append(results)

            tracked_loss = self._train_epoch(
                debug=debug,
                transformation=None,
                use_ground_truth=use_ground_truth,
                dataloader=dataloader
            )
            print(tracked_loss)
            wandb.log({"linear-loss": tracked_loss})
            # print(torch.sigmoid(self.c_params) > 0.5)

            if use_scheduler:
                scheduler.step(tracked_loss)

        # if eval_interval is not None:
        #     # Eval at end of training
        #     final_results = self.eval_A(
        #         observations=dataset.o_supportx,
        #         latents=dataset.latents,
        #         transformation=dataset.transformation,
        #         epoch=(num_epochs - 1),
        #         causal_index=dataset.causal_index,
        #         true_gammas=dataset.gamma_coeffs,
        #         approx_gammas=self.gamma_params,
        #         use_ground_truth=use_ground_truth,
        #     )
        #     final_results['epoch'] = num_epochs - 1
        #     train_results.append(final_results)

        #     results_df = pd.DataFrame.from_records(train_results)
        # else:
        #     results_df = None
        results_df = None
        return wandb.run.name, wandb.run.id, results_df
    
    def get_features(self, dataset, run_id=None, batch_size=1024):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        z = []
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                z_batch = batch[0].to(self.device) @ self.A.weight.data.inverse().T
                z.append(z_batch)
            z = torch.cat(z, dim=0).detach().cpu().numpy()

        if run_id is not None:
            np.save("./experiments/qm9/latents/z{}_{}.npy".format(run_id, self.latent_dim), z)

        return z

    @staticmethod
    def _init_free_params(size, fixed_value=None, warmup=False):
        if fixed_value is not None:
            if not warmup:
                free_params = fixed_value * torch.ones(size)
            else:  # In this case, the values are fixed initially but will be unfrozen later
                free_params = torch.nn.Parameter(
                    fixed_value * torch.ones(size, dtype=torch.float32)
                )
        else:
            free_params = torch.nn.Parameter(torch.randn(size, dtype=torch.float32))
        return free_params

    def _train_epoch(
        self,
        debug,
        transformation,
        use_ground_truth,
        dataloader,
    ):

        """
        Traverses through the batches and performs the training for one full epoch.
        """
        self.train()
        epochs_losses = []
        with tqdm(dataloader) as T:
            for batch_idx, data_batch in enumerate(T):

                inputs, targets, task_idx_batch = data_batch
                # batch_size = inputs.shape[0]

                # inputs = inputs.unsqueeze(1).expand(-1, self.num_tasks, -1).reshape(batch_size*self.num_tasks, self.observation_dim)
                # targets = targets.unsqueeze(-1).reshape(batch_size*self.num_tasks, 1)
                # gamma = self.gamma_params.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size*self.num_tasks, self.observation_dim).unsqueeze(-1)
                # c_s = self.c_params.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size*self.num_tasks, self.observation_dim).unsqueeze(1)
                gamma = self.gamma_params[task_idx_batch].unsqueeze(-1)
                c_s = self.c_params[task_idx_batch].unsqueeze(1)

                self.optimizer.zero_grad()
                res = self.likelihood_loss(
                    inputs.to(self.device),
                    targets.to(self.device),
                    causal_index=None,
                    transformation=transformation,
                    use_ground_truth=use_ground_truth,
                    debug=debug,
                    c_ind_params=c_s,
                    gamma=gamma,
                    true_gammas=None,
                )
                if len(res) == 4:
                    loss, Mu, Sigma, c_indx = res
                else:
                    loss, Mu, Sigma, gt_loss, c_indx = res
                loss = loss.mean()
                epochs_losses.append(loss.item())
                loss.backward()

                # if len(res) == 5:
                #     wandb.log({"linear-loss": loss.item(), "linear-gt_loss": gt_loss.mean().item()})
                # else:
                #     wandb.log({"linear-loss": loss.item()})

                self.optimizer.step()
                # wandb.log(
                #     {"sigma_s": self.sigma_s.item(), "sigma_obs": self.sigma_obs.item()}
                # )  # Log noise variables

                # Update progress bar
                if batch_idx % 50 == 0:
                    T.set_postfix(dict(loss=float(loss.item())))

        return np.mean(epochs_losses)

    def eval_A(
        self,
        observations,
        epoch,
        latents=None,
        transformation=None,
        causal_index=None,
        true_gammas=None,
        approx_gammas=None,
        use_ground_truth=False,
    ):
        """
        Evaluate the current A matrix via the MCC score with respect to both the ground truth transformation
        and the original latent representation (which includes the noise).

        Args:
            observations: Observations from the data. Shape is (num_tasks, num_samples, observation_dim).
            latents: Latent representation of the data. Shape is (num_tasks, num_samples, latent_dim).
            transformation: Ground truth transformation matrix. Shape is (latent_dim, observation_dim).
            epoch: Current epoch.

        Returns:
            results_dict: Dictionary containing the MCC scores for the transformation and the latent representation.
        """
        self.eval()

        if latents is not None or transformation is not None:
            with torch.no_grad():
                mccs_transforms = []
                mccs_latents = []
                mccs_latents_transforms = []
                assignments_transforms = []
                assignments_latents = []
                assignments_latents_transforms = []
                with torch.no_grad():
                    if observations.shape[1] > 1:
                        z_hat = observations @ self.A.weight.data.inverse().T
                        if transformation is not None:
                            z = observations @ transformation.inverse().T
                        for t in range(latents.shape[0]):
                            if transformation is not None:
                                mcc_transform, assignments_transform = cal_mcc(
                                    z[t].cpu().detach().numpy(),
                                    z_hat[t].cpu().detach().numpy(),
                                    pearson=False,
                                )
                                mccs_transforms.append(mcc_transform)
                                assignments_transforms.append(assignments_transform)
                            if latents is not None:
                                mcc_latent, assignments_latent = cal_mcc(
                                    latents[t].cpu().detach().numpy(),
                                    z_hat[t].cpu().detach().numpy(),
                                    pearson=False,
                                )
                                mccs_latents.append(mcc_latent)
                                assignments_latents.append(assignments_latent)
                            if transformation is not None and latents is not None:
                                mcc_latents_transforms, assignments_gt = cal_mcc(
                                    latents[t].cpu().detach().numpy(),
                                    z[t].cpu().detach().numpy(),
                                    pearson=False,
                                )
                                mccs_latents_transforms.append(mcc_latents_transforms)
                                assignments_latents_transforms.append(assignments_gt)

                        if self.log:
                            print(f"Mean MCC: {np.mean(mccs_transforms)}")
                        wandb.log(
                            {"MCC_transform": np.mean(mccs_transforms), "epoch": epoch}
                        )
                        wandb.log(
                            {"MCC_true_latents": np.mean(mccs_latents), "epoch": epoch}
                        )
                        wandb.log(
                            {
                                "MCC_true_latents_transform": np.mean(
                                    mccs_latents_transforms
                                ),
                                "epoch": epoch,
                            }
                        )

                        if (
                            not use_ground_truth
                            and latents is not None
                            and causal_index is not None
                        ):
                            self._check_causal_index_recovery(
                                assignments_latents, causal_index, epoch
                            )
                        if true_gammas is not None and approx_gammas is not None:
                            if use_ground_truth:
                                # MSE should be zero
                                self._check_gamma_convergence(
                                    assignments_latents_transforms,
                                    true_gammas=true_gammas,
                                    approx_gammas=true_gammas,
                                    epoch=epoch,
                                )
                            else:
                                self._check_gamma_convergence(
                                    assignments_latents,
                                    true_gammas=true_gammas,
                                    approx_gammas=approx_gammas,
                                    epoch=epoch,
                                )

                    self._console_log(assignments_latents, causal_index, transformation)
        else:
            raise ValueError(
                "At least one of latents and transformation must be provided."
            )

        results_dict = {
            "mcc_latents_mean": np.mean(mccs_latents),
            "mcc_latents_transforms_mean": np.mean(mccs_latents_transforms),
            "mcc_transforms_mean": np.mean(mccs_transforms),
        }
        return results_dict

    def _check_gamma_convergence(
        self, assignments_latents, true_gammas, approx_gammas, epoch
    ):
        """
        Check the convergence of the gamma parameters. First determines the permutation of the latent variables according
        to the linear sum assignment and inverts the permutation to recover the original latent ordering. Then assess the MSE
        between the true and approximated gamma parameters after the inverted permutation.

        Args:
            assignment_latents: List of tuples mapping the latent index (first tuple) to the observed index (second tuple).
            true_gammas: True values for the gamma parameters.
            approx_gammas: Approximated values for the gamma parameters.
            epoch: Current epoch.

        Returns:
            None (logs to wandb)

        """
        permutation_index = torch.tensor(np.array(assignments_latents))[:, 1]
        permuted_approx_gammas = self._permute_tensor_columns(
            approx_gammas, permutation_index
        )
        with torch.no_grad():
            gamma_mse = torch.mean((permuted_approx_gammas - true_gammas) ** 2)
            wandb.log({"gamma_mse": gamma_mse.item(), "epoch": epoch})

    def _check_causal_index_recovery(self, assignments_latents, causal_index, epoch):
        """
        Check the accuracy of the causal index recovery. First determines the permutation of the latent variables according
        to the linear sum assignment and inverts the permutation to recover the original latent ordering. Then, the causal
        index is recovered by indexing into the causal index parameters with the recovered latent ordering.

        Args:
            assignments_latents: List of tuples mapping the latent index (first tuple) to the observed index (second tuple).
            causal_ind_batches: Ground truth values for the causal index.
            epoch: Current epoch.

        Returns:

        """

        permutation_index = torch.tensor(np.array(assignments_latents))[:, 1]
        permuted_c_index = self._permute_tensor_columns(
            torch.sigmoid(self.c_params), permutation_index
        )
        c_index_accuracy = (
            ((permuted_c_index > 0.5) == causal_index).all(dim=1).float().mean()
        )
        wandb.log({"c_index_accuracy": c_index_accuracy.item(), "epoch": epoch})

    def _permute_tensor_columns(self, inputs, permutation_index):
        """
        Permute the columns of a tensor row by row according to the permutation index.

        Args:
            inputs: Input data, shape is (num_tasks, permutation dimension).
            permutation_index: Permutation index, shape is (num_tasks, permutation dimension).

        Returns:

        """
        permuted_inputs = torch.stack(
            [inputs[i, permutation_index[i]] for i in range(len(inputs))]
        )
        return permuted_inputs

    def _console_log(self, assignments_latents, causal_index, transformation):
        if self.log:
            print(f"A_approx: {self.A.weight.data}")
            print(f"A_true: {transformation}")
            rand_elem_ix = np.random.randint(0, len(self.c_params))
            print(f"c_index_approx: {torch.sigmoid(self.c_params[rand_elem_ix])}")
            print(f"c_index_gt: {causal_index[rand_elem_ix]}")

            print(
                f" Linear assignment of true latents to approx latents: {assignments_latents[rand_elem_ix]}"
            )


if __name__ == "__main__":
    import argparse

    # Process command line arguments for num_epochs, num_tasks, num_causal, observation_dim, fixed_gamma, warmup
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--observation_dim", type=int, default=5)
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--fixed_gamma", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    observation_dim = args.observation_dim
    fixed_gamma = args.fixed_gamma
    warmup = args.warmup
    run_id = args.run_id
    batch_size = args.batch_size

    x = np.load("./experiments/qm9/latents/h{}_{}.npy".format(run_id, observation_dim))
    y = np.genfromtxt("./experiments/qm9/data/y_all.csv", delimiter=",")
    num_tasks = y.shape[-1]

    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    dataset = TensorDataset(x)

    task_id = torch.tensor([i for i in range(num_tasks) for _ in range(x.shape[0])]).long()
    x_task = torch.cat([x for _ in range(num_tasks)], dim=0)
    y_task = torch.cat([y[:, i] for i in range(num_tasks)], dim=0).unsqueeze(-1)
    dataset_task = TensorDataset(x_task, y_task, task_id)

    wandb.init(project="causal-ml", name="task-linear-model", group="qm9")
    model = TaskLinearModel(
        observation_dim=observation_dim,
        latent_dim=observation_dim,
        num_tasks=num_tasks,
        amortized=False,
        device="cpu",
        sigma_s=None,
        sigma_obs=0.01,
    )

    model.train_A(
        dataset=dataset_task,
        num_epochs=num_epochs,
        debug=True,
        use_ground_truth=False,
        batch_size=batch_size, 
        optimizer=None,  # partial(torch.optim.LBFGS, max_iter=500),
        eval_interval=None,
        fixed_gamma=fixed_gamma,
        warmup_epochs=warmup,
        use_scheduler=True,
    )

    model.get_features(dataset, run_id, batch_size=1024)