import os
import pandas as pd
import torch
import datamol as dm
from itertools import product
from mtlcm.data.synthetic.linear import LinearDataset
from mtlcm.utils.data.generics import seed_everything
from mtlcm.models.linear_task_model import TaskLinearModel

class LinearTransformExperiment:
    def __init__(self, output_path, latent_dim, num_seeds=1, matrix_types=None, sigma_obs=0.01, sigma_s=0.1,
                 num_causal=2, num_tasks=100, num_points_per_task=50, num_epochs=6000,
                 batch_size=20, device=None, n_jobs=1, standardize_features=True,
                 seed=None):

        self.standardize_features = standardize_features
        self.output_path = output_path
        self.seeds = [seed] if seed is not None else list(range(num_seeds))
        self.latent_dim = latent_dim
        self.matrix_types = matrix_types
        self.sigma_obs = sigma_obs
        self.sigma_s = sigma_s
        self.num_causal = num_causal
        self.num_tasks = num_tasks
        self.num_points_per_task = num_points_per_task
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_jobs = n_jobs

    def run(self):
        """
        Run the experiment.

        Args:
            save_path: str
                Path to save the results of the experiment.
            seed: int
                Random seed.
            latent_dim: int
                Dimension of the latent factors of variation (= to observation dim).
            matrix_type: str
                Type of matrix to use for the linear transformation. Options are "identity", "orthogonal", or None.

        Returns:
            pd.DataFrame containing the results of the experiment.

        """
        kwargs = [{'seed': x[0], 'matrix_type': x[1]} for x in list(product(self.seeds, self.matrix_types))]
        results_list = dm.utils.parallelized(fn=self._run_identifiability_experiment, inputs_list=kwargs,
                                             n_jobs=self.n_jobs, arg_type="kwargs", progress=True,
                                             scheduler="threads")

        results = pd.concat(results_list, axis=0) 
        self._save_results(results, save_path=os.path.join(self.output_path, "results.csv"))

    def setup(self, seed, matrix_type, save_data=True):
        """
        Generate the ground truth data and decoder for the experiment and serialize it to disk.

        Returns: LinearDataset

        """
        if matrix_type == "identity":
            identity = True
            orthogonal = False
        elif matrix_type == "orthogonal":
            identity = False
            orthogonal = True
        else:
            identity = False
            orthogonal = False

        dataset = LinearDataset(
            observation_dim=self.latent_dim,
            num_tasks=self.num_tasks,
            num_support_points=self.num_points_per_task,
            sigma_obs=self.sigma_obs,
            sigma_s=self.sigma_s,
            identity=identity,
            orthogonal=orthogonal,
            num_causal=self.num_causal,
            standardize_features=self.standardize_features,
            device=self.device,
        )

        # Save the data
        if save_data:
            save_path = os.path.join(self.output_path, f"seed_{str(seed)}", "data")
            os.makedirs(save_path, exist_ok=True)
            torch.save(dataset, os.path.join(save_path, "dataset.pt"))

        return dataset

    @staticmethod
    def _save_results(results, save_path):
        """
        Save the results dataframe in save_path. Will add any tags as columns to the dataframe.

        Args:
            results: pd.DataFrame
            save_path: str
            tags: dict

        Returns:

        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # If the file already exists, load it and append the new results
        if os.path.exists(save_path):
            old_results = pd.read_csv(save_path)
            results = pd.concat([old_results, results], axis=0)

        results.to_csv(save_path, index=False)

    @staticmethod
    def _history_to_df(results):
        rows = [r for r in results]
        results = pd.DataFrame(rows)
        return results

    def _run_identifiability_experiment(self, seed, matrix_type):

        seed_everything(seed)
        dataset = self.setup(seed, matrix_type)

        r = self._train_model(
            dataset=dataset,
            device=self.device,
        )

        # annotate the results with settings
        r["matrix_type"] = matrix_type
        r["seed"] = seed
        r["latent_dim"] = self.latent_dim
        r["num_causal"] = self.num_causal
        
        return r

    def _train_model(
        self,
        dataset,
        device=None,
    ):
        """
        Generates the data for the synthetic experiments, trains the model with the given inputs and targets and using
        the given parameters for the ground truth values.

        Args:
            inputs:
            targets:
            use_ground_truth:

        Returns:

        """

        model = TaskLinearModel(
            observation_dim=self.latent_dim,
            latent_dim=self.latent_dim,
            device=device,
        )

        run_results = model.train_A(
            dataset=dataset,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_scheduler=True,
        )

        return run_results


