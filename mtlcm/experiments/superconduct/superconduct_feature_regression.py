import os
import fsspec
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from loguru import logger
from torch.utils.data import TensorDataset
from mtlcm.models import MultiTaskModel, TaskLinearModel
from mtlcm.utils.data.generics import seed_everything

DATA_PATH = Path(__file__).parent.parent.parent / "data/superconduct"

class SuperconductFeatureRegressionExperiment:
    def __init__(
        self,
        output_path,
        observation_dim,
        latent_dim,
        hidden_dim,
        device=None,
        seed=0,
        sigma_obs=1e-4,
        sigma_s=0.1,
        num_linear_epochs=20,
        num_multitask_epochs=200,
        batch_size=256,
        exp_name=None,
        standardize_features=True,
        lr=1e-3,
        feature_regression=False,
    ):

        self.feature_regression = feature_regression
        self.sigma_obs = sigma_obs
        self.standardize_features = standardize_features
        self.sigma_s = sigma_s
        self.exp_name = "" if exp_name is None else exp_name

        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.output_path = os.path.join(output_path, f"seed_{seed}")
        self.hidden_dim = hidden_dim
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_linear_epochs = num_linear_epochs
        self.num_multitask_epochs = num_multitask_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.lr = lr
        self.num_tasks = None

    def setup(self, seed):
        """
        Generate the ground truth data and decoder for the experiment.

        Returns: None

        """
        seed_everything(seed)

        x = np.genfromtxt(
            os.path.join(DATA_PATH, "unique_m.csv"), skip_header=1, delimiter=","
        )[
            :, :-1
        ]  # the last column is string, which we skip
        y = np.genfromtxt(
            os.path.join(DATA_PATH, "train.csv"), skip_header=1, delimiter=","
        )[
            :, 1:
        ]  # the first column is the number of elements, which is trivial and skipped

        # Standardize the features
        if self.standardize_features:
            x, y = self._standardize_data(x, y)

        dataset = TensorDataset(x, y)
        self.num_tasks = y.shape[-1]
        self.x = x
        self.y = y

        return dataset

    def _standardize_data(self, *tensors):
        return [
            torch.from_numpy(StandardScaler().fit_transform(t)).to(self.device).float()
            for t in tensors
        ]

    def run(self):
        """
        The experiment consists of two stages. In the first stage, we train the multitask model in order to obtain a representation which has a linear
        relationship to the ground truth latent variables. In the second stage, we train the identifiable linear model on top of this representation to obtain
        a representation which should be identifiable up to permutations and scaling.

        Returns:
            None (Results are saved to disk)
        """

        # Train the multitask model
        logger.info("Training the multitask model")
        multi_dataset = self.setup(seed=self.seed)
        self._train_multitask(dataset=multi_dataset)

        # Train the identifiable linear model
        logger.info("Training the identifiable linear model")
        observations, _ = self._standardize_data(*self.multitask_model.get_latents(multi_dataset))
        with fsspec.open(
            f"{self.output_path}/data/multitask_model_latents.pt", "wb"
        ) as f:
            torch.save(observations, f)

        task_id = torch.tensor(
            [i for i in range(self.num_tasks) for _ in range(observations.shape[0])]
        ).long()
        x_task = torch.cat([observations for _ in range(self.num_tasks)], dim=0)
        y_task = torch.cat(
            [self.y[:, i] for i in range(self.num_tasks)], dim=0
        ).unsqueeze(-1)
        lin_model_dataset = TensorDataset(x_task, y_task, task_id)

        results = self._train_linear_model(dataset=lin_model_dataset)
        lin_latents = self.linear_model.get_latents(observations, to_numpy=False)
        with fsspec.open(f"{self.output_path}/data/linear_model_latents.pt", "wb") as f:
            torch.save(lin_latents, f)

        ## Save results
        self.save_results(results=results)

    def _train_linear_model(self, dataset):

        dataset.num_tasks = self.num_tasks
        dataset.transformation = None

        self.linear_model = TaskLinearModel(
            observation_dim=self.latent_dim,
            latent_dim=self.latent_dim,
            device=self.device,
            sigma_obs=self.sigma_obs,
        )

        results_df = self.linear_model.train_A(
            dataset=dataset,
            num_epochs=self.num_linear_epochs,
            batch_size=self.batch_size,
            eval_interval=200,
            use_scheduler=True,
            run_eval=False,
        )

        results_df["seed"] = self.seed
        results_df["num_tasks"] = self.num_tasks
        results_df["latent_dim"] = self.latent_dim
        results_df["sigma_obs"] = self.sigma_obs
        results_df["observed_dim"] = self.observation_dim

        return results_df

    def _train_multitask(self, dataset):

        self.multitask_model = MultiTaskModel(
            observation_dim=self.observation_dim,
            latent_dim=self.latent_dim,
            num_tasks=self.num_tasks,
            hidden_dim=self.hidden_dim,
            device=self.device,
        )

        self.multitask_model.train_predictor(
            dataset=dataset,
            num_epochs=self.num_multitask_epochs,
            batch_size=self.batch_size,
            use_scheduler=False,
            lr=self.lr,
        )

    def save_results(self, results):
        save_path = f"{self.output_path}/results.csv"
        results.to_csv(save_path)
