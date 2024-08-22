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
        gamma,
        c_ind_params,
    ):
        """
        Compute the MLE objective for the linear model.

        Args:
            inputs: Input observations. Shape is (num tasks x num samples x num observed features)
            targets: Target observations. Shape is (num tasks x num samples x 1)
            c_ind_params: Parameters for the causal indices. Shape is (num tasks x 1 x num latent features)
            gamma: Spurious variable correlation parameter. Shape is (num_tasks x num latent features x 1)

        Returns:
            torch.float: negative log-likelihood value
        """

        c_indx = torch.sigmoid(c_ind_params)

        Mu, Sigma = self._get_distribution_params(
            causal_index=c_indx,
            inputs=inputs,
            targets=targets,
            transformation=self.A,
            gamma=gamma,
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
        batch_size=32,
        optimizer=None,
        eval_interval=50,
        warmup_epochs=0,
        use_scheduler=False,
        run_eval=True,
    ):
        """
        Given observed inputs and outputs, train the linear transformation matrix A to recover the ground truth transformation
        matrix of the latent space.

        Args:
            dataset: LinearDataset object.
            num_epochs: Number of epochs to train for.
            latents: Latent variables representating the underlying factors of variation. Shape is (num tasks x num samples x num latent features).
            batch_size: Batch size of tasks for training.
            warmup_epochs: Number of epochs at start of training during which the parameters for gamma are frozen. After this point, everything is trained jointly.

        Returns:

        """

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.c_params = self._init_free_params(
            size=(dataset.num_tasks, self.observation_dim)
        )
        self.gamma_params = self._init_free_params(
            size=(dataset.num_tasks, self.observation_dim),
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
        with tqdm(range(num_epochs)) as T:
            for epoch in T:
                if 0 < warmup_epochs == epoch:
                    unfreeze(self.gamma_params)
                elif warmup_epochs > 0 and epoch < warmup_epochs:
                    freeze(self.gamma_params)

                # Eval progress every eval_interval epochs
                if (epoch % eval_interval) == 0 and run_eval:
                    results = self.eval_A(
                        observations=dataset.o_supportx.to(self.device),
                        latents=(
                            dataset.latents.to(self.device)
                            if dataset.latents is not None
                            else None
                        ),
                        transformation=(
                            dataset.transformation.to(self.device)
                            if dataset.transformation is not None
                            else None
                        ),
                    )
                    results["linear_model_epoch"] = epoch
                    train_results.append(results)

                tracked_loss = self._train_epoch(
                    dataloader=dataloader,
                )

                if use_scheduler:
                    scheduler.step(tracked_loss)

                T.set_postfix(train_loss=tracked_loss)

        # Eval at end of training
        if run_eval:
            final_results = self.eval_A(
                observations=dataset.o_supportx.to(self.device),
                latents=(
                    dataset.latents.to(self.device)
                    if dataset.latents is not None
                    else None
                ),
                transformation=(
                    dataset.transformation.to(self.device)
                    if dataset.transformation is not None
                    else None
                ),
            )
            final_results["linear_model_epoch"] = num_epochs - 1
            train_results.append(final_results)

        results_df = pd.DataFrame.from_records(train_results)
        return results_df

    @staticmethod
    def _init_free_params(size):
        free_params = torch.nn.Parameter(torch.randn(size, dtype=torch.float32))
        return free_params

    def _train_epoch(
        self,
        dataloader,
    ):
        """
        Traverses through the batches and performs the training for one full epoch.
        """
        self.train()
        epochs_losses = []

        for _, data_batch in enumerate(dataloader):

            inputs, targets, task_idx_batch = data_batch
            gamma = self.gamma_params[task_idx_batch].unsqueeze(-1)
            c_s = self.c_params[task_idx_batch].unsqueeze(1)

            # Put on device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            loss, _, _, _ = self.likelihood_loss(
                inputs,
                targets,
                c_ind_params=c_s,
                gamma=gamma,
            )

            loss = loss.mean()
            epochs_losses.append(loss.item())
            loss.backward()

            self.optimizer.step()

        return np.mean(epochs_losses)

    def eval_A(
        self,
        observations,
        latents=None,
        transformation=None,
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
