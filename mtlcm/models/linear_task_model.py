from loguru import logger
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from mtlcm.utils.data.lin_transform import cal_mcc
from mtlcm.utils.data.generics import freeze, unfreeze
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TaskLinearModel(nn.Module):
    def __init__(
        self,
        observation_dim,
        latent_dim,
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
            sigma_s: Standard deviation of the spurious noise. If None, this is learned as an additional parameter (not recommended).
        """

        super(TaskLinearModel, self).__init__()
        self.log = log
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
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

    def get_latents(self, x, batch_size=1024, to_numpy=True):
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        z = []
        self.eval()
        with torch.no_grad():
            for data in dataloader:
                z_batch = data[0].to(self.device) @ self.A.weight.data.inverse().T
                z.append(z_batch)
            z = torch.cat(z, dim=0).detach()

        if to_numpy:
            z = z.cpu().numpy()

        return z

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
        try:
            log_prob = self._likelihood(Mu, Sigma, inputs)
        except ValueError:
            # Add small amount of noise to the diagonal to ensure that the covariance matrix is positive definite
            logger.warning(
                "Covariance matrix was not positive definite. Adding epsilon of 1e-6 to the diagonal."
            )
            Sigma = Sigma + 1e-6 * torch.eye(Sigma.shape[-1]).to(self.device)
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

        D = torch.diag_embed(
            self.sigma_s**2 * (1 - causal_index) + causal_index
        ).squeeze(1)
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
        eval_interval=50,
        fixed_gamma=None,
        warmup_epochs=0,
        use_scheduler=False,
        dtype=torch.float32,
        run_eval=True,
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
            size=(dataset.num_tasks, self.observation_dim), fixed_value=None
        )
        self.gamma_params = self._init_free_params(
            size=(dataset.num_tasks, self.observation_dim),
            fixed_value=fixed_gamma,
            warmup=(warmup_epochs > 0),
        )

        if optimizer is None:
            lr = 0.1 if use_scheduler else 5e-3
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(
                self.parameters()
            )  # set any extra parameters as a partial function object (functools)

        self.to(self.device)

        if use_scheduler:
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                "min",
                patience=100,
                factor=0.5,
                verbose=True,
                min_lr=5e-3,
                threshold=1e-4,
            )

        train_results = []
        for epoch in range(num_epochs):
            if 0 < warmup_epochs == epoch:
                unfreeze(self.gamma_params)
            elif warmup_epochs > 0 and epoch < warmup_epochs:
                freeze(self.gamma_params)

            # Eval progress every eval_interval epochs
            if (epoch % eval_interval) == 0 and run_eval:
                results = self.eval_A(
                    observations=dataset.o_supportx.to(self.device),
                    latents=dataset.latents.to(self.device)
                    if dataset.latents is not None
                    else None,
                    transformation=dataset.transformation.to(self.device)
                    if dataset.transformation is not None
                    else None,
                    epoch=epoch,
                    causal_index=dataset.causal_index.to(self.device)
                    if dataset.causal_index is not None
                    else None,
                    true_gammas=dataset.gamma_coeffs.to(self.device)
                    if dataset.gamma_coeffs is not None
                    else None,
                    approx_gammas=self.gamma_params,
                    use_ground_truth=use_ground_truth,
                )
                results["linear_model_epoch"] = epoch
                train_results.append(results)

            tracked_loss = self._train_epoch(
                debug=debug,
                transformation=dataset.transformation.to(self.device)
                if dataset.transformation is not None
                else None,
                use_ground_truth=use_ground_truth,
                dataloader=dataloader,
            )

            if use_scheduler:
                scheduler.step(tracked_loss)

        # Eval at end of training
        if run_eval:
            final_results = self.eval_A(
                observations=dataset.o_supportx.to(self.device),
                latents=dataset.latents.to(self.device)
                if dataset.latents is not None
                else None,
                transformation=dataset.transformation.to(self.device)
                if dataset.transformation is not None
                else None,
                epoch=(num_epochs - 1),
                causal_index=dataset.causal_index.to(self.device)
                if dataset.causal_index is not None
                else None,
                true_gammas=dataset.gamma_coeffs.to(self.device)
                if dataset.gamma_coeffs is not None
                else None,
                approx_gammas=self.gamma_params,
                use_ground_truth=use_ground_truth,
            )
            final_results["linear_model_epoch"] = num_epochs - 1
            train_results.append(final_results)

        results_df = pd.DataFrame.from_records(train_results)
        results_df

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
                if len(data_batch) == 5:
                    inputs, targets, causal_index, true_gammas, task_idx_batch = (
                        data_batch
                    )
                else:
                    inputs, targets, task_idx_batch = data_batch
                    causal_index = None
                    true_gammas = None

                gamma = self.gamma_params[task_idx_batch].unsqueeze(-1)
                c_s = self.c_params[task_idx_batch].unsqueeze(1)

                # Put on device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                res = self.likelihood_loss(
                    inputs,
                    targets,
                    causal_index=causal_index,
                    transformation=transformation,
                    use_ground_truth=use_ground_truth,
                    debug=debug,
                    c_ind_params=c_s,
                    gamma=gamma,
                    true_gammas=true_gammas,
                )
                if len(res) == 4:
                    loss, Mu, Sigma, c_indx = res
                else:
                    loss, Mu, Sigma, gt_loss, c_indx = res
                loss = loss.mean()
                epochs_losses.append(loss.item())
                loss.backward()

                self.optimizer.step()

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


# if __name__ == "__main__":
#     import argparse
#     # from data.synthetic.linear import LinearDataset

#     # Process command line arguments for num_epochs, num_tasks, num_causal, observation_dim, fixed_gamma, warmup
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_epochs", type=int, default=100)
#     parser.add_argument("--num_tasks", type=int, default=100)
#     parser.add_argument("--num_causal", type=int, default=2)
#     parser.add_argument("--observation_dim", type=int, default=10)
#     parser.add_argument("--fixed_gamma", type=float, default=None)
#     parser.add_argument("--warmup", type=int, default=0)

#     args = parser.parse_args()
#     num_epochs = args.num_epochs
#     num_tasks = args.num_tasks
#     num_causal = args.num_causal
#     observation_dim = args.observation_dim
#     fixed_gamma = args.fixed_gamma
#     warmup = args.warmup

#     # dataset = LinearDataset(observation_dim=observation_dim, num_tasks=num_tasks, num_causal=num_causal,
#     #                      sigma_obs=0.01, sigma_s=0.1, sample_gammas=True, num_support_points=50,
#     #                      identity=False, orthogonal=False, device="cpu")

#     model = TaskLinearModel(
#         observation_dim=observation_dim,
#         latent_dim=observation_dim,
#         amortized=False,
#         device="cpu",
#     )

#     model.train_A(
#         dataset=dataset,
#         num_epochs=num_epochs,
#         debug=True,
#         use_ground_truth=False,
#         batch_size=256,
#         optimizer=None,  # partial(torch.optim.LBFGS, max_iter=500),
#         eval_interval=50,
#         fixed_gamma=fixed_gamma,
#         warmup_epochs=warmup,
#         use_scheduler=True,
#     )
