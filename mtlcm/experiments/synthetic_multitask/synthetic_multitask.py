import os
import glob
import fsspec
import torch
import torch.nn as nn
from loguru import logger
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mtlcm.models import MultiTaskModel
from mtlcm.models import TaskLinearModel
from mtlcm.utils.data.generics import seed_everything
from mtlcm.data.synthetic.non_linear import NonLinearDataset
from mtlcm.data.synthetic.linear import LinearDataset


class SyntheticMultiTaskExperiment:
    def __init__(
        self,
        output_path,
        observation_dim,
        latent_dim,
        num_tasks,
        num_samples_per_task,
        num_causal,
        hidden_dim,
        num_hidden_layers: int = 1,
        last_dim=None,
        device=None,
        seed=0,
        sigma_obs=1e-4,
        num_linear_epochs=1000,
        num_multitask_epochs=2000,
        batch_size=256,
        exp_name=None,
        standardize_features=True,
    ):

        self.sigma_obs = sigma_obs
        self.standardize_features = standardize_features
        self.num_samples_per_task = num_samples_per_task
        self.exp_name = "" if exp_name is None else exp_name
        self.num_tasks = num_tasks
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.output_path = os.path.join(output_path, f"seed_{seed}")
        self.hidden_dim = hidden_dim
        self.last_dim = last_dim
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_causal = num_causal
        self.num_hidden_layers = num_hidden_layers
        self.num_linear_epochs = num_linear_epochs
        self.num_multitask_epochs = num_multitask_epochs
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, seed, save_data=True):
        """
        Generate the ground truth data and decoder for the experiment.

        Returns: None

        """
        seed_everything(seed)
        self.true_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.observation_dim),
            nn.ReLU(),
            nn.Linear(self.observation_dim, self.observation_dim),
        )

        # Generate the data
        dataset = NonLinearDataset(
            decoder=self.true_decoder, observation_dim=self.observation_dim, latent_dim=self.latent_dim,
            sigma_s=0.1, num_causal=self.num_causal, num_tasks=self.num_tasks, device=self.device,
            sample_gammas=True, num_support_points=self.num_samples_per_task, standardize_features=self.standardize_features)

        # Save artifacts
        if save_data:
            save_path = os.path.join(self.output_path, "data")
            os.makedirs(save_path, exist_ok=True)
            torch.save(dataset, os.path.join(save_path, "dataset.pt"))

        return dataset

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
        _, weak_mcc = self._train_multitask(dataset=multi_dataset)

        observations = self.multitask_model.get_representation_from_ground_truth(
            x=multi_dataset.x_data.to(self.device),
        )
        with fsspec.open(f"{self.output_path}/data/observations.pt", "wb") as f:
            torch.save(observations, f)
            
        # Train the identifiable linear model
        logger.info("Training the identifiable linear model")
        linear_dataset = LinearDataset.from_data(o_supportx=observations, o_supporty=multi_dataset.y_data, num_tasks=self.num_tasks,
                                                 num_support_points=self.num_samples_per_task, device=self.device, causal_index=multi_dataset.causal_index,
                                                 gamma_coeffs=multi_dataset.gamma_coeffs, latents=multi_dataset.x_data)

        results = self._train_linear_model(dataset=linear_dataset)
        
        # These are the results from the multitask (weakly identifiable) model
        results["weak_mcc_after_training"] = weak_mcc

        # Save results
        self.save_results(results=results)

    def _train_linear_model(self, dataset):

        linear_model = TaskLinearModel(
            observation_dim=self.latent_dim,
            latent_dim=self.latent_dim,
            device=self.device,
            sigma_obs=self.sigma_obs,
        )

        results_df = linear_model.train_A(
            dataset=dataset,
            num_epochs=self.num_linear_epochs,
            debug=True,
            use_ground_truth=False,
            batch_size=self.batch_size,
            eval_interval=200,
            use_scheduler=True,
        )
        
        results_df['seed'] = self.seed
        results_df['num_causal'] = self.num_causal
        results_df['num_tasks'] = self.num_tasks
        results_df['num_samples_per_task'] = self.num_samples_per_task
        results_df['latent_dim'] = self.latent_dim
        results_df['sigma_obs'] = self.sigma_obs
        results_df['observed_dim'] = self.observation_dim


        return results_df

    def _train_multitask(self, dataset):

        self.multitask_model = MultiTaskModel(
            observation_dim=self.observation_dim,
            latent_dim=self.latent_dim,
            num_tasks=self.num_tasks,
            true_decoder=self.true_decoder,
            hidden_dim=self.hidden_dim,
            last_dim=self.last_dim,
            num_hidden_layers=self.num_hidden_layers,
            device=self.device,
        )

        self.multitask_model.train_predictor(
            dataset=dataset,
            num_epochs=self.num_multitask_epochs,
            batch_size=self.batch_size,
            use_scheduler=False,
        )

        strong_mcc, weak_mcc = self.multitask_model.eval_mcc(
            observations=dataset.obs_data, targets=dataset.x_data.detach().cpu().numpy(), cca_dim=self.latent_dim)
        
        return strong_mcc, weak_mcc

    def save_results(self, results):
        save_path = f"{self.output_path}/results.csv"
        results.to_csv(save_path)



