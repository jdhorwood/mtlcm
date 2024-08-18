import os
import glob
import fsspec
import torch
import yaml
import numpy as np
import itertools
from dgl.dataloading.dataloader import GraphDataLoader
from matplotlib import pyplot as plt
from loguru import logger
import pandas as pd
from torch.utils.data import TensorDataset
import seaborn as sns
from mtlcm.models import MultiTaskModel, TaskLinearModel
from mtlcm.utils.data.generics import seed_everything, standardize_data
from mtlcm.utils.data.lin_transform import cal_weak_strong_mcc, cal_mcc
from mtlcm.data.qm9.graph_dataset import GraphDataset

DATA_PATH = "./data/qm9"

class QM9Experiment:
    def __init__(
            self,
            output_path,
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
            use_gnn=True,
            load_path=None,
            subset_size=None,
    ):

        self.load_path = load_path
        self.subset_size = subset_size
        self.use_gnn = use_gnn
        self.sigma_obs = sigma_obs
        self.standardize_features = standardize_features
        self.sigma_s = sigma_s
        self.exp_name = "" if exp_name is None else exp_name

        self.latent_dim = latent_dim
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

        # Load the data
        if self.use_gnn:
            dataset = GraphDataset(subset_size=self.subset_size, standardize=True, load_path=self.load_path)
            self.y = dataset.y
            self.observation_dim = dataset.num_features
            self.num_tasks = dataset.num_tasks
        else:
            x = np.genfromtxt(os.path.join(DATA_PATH, "e_all.csv"), delimiter=",")
            y = np.genfromtxt(os.path.join(DATA_PATH, "y_all.csv"), delimiter=",")
            self.observation_dim = x.shape[-1]
            self.num_tasks = y.shape[-1]

            # Standardize the features
            if self.standardize_features:
                x, y = standardize_data(x, y, device=self.device)

            dataset = TensorDataset(x, y)
            self.x = x
            self.y = y

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
        self._train_multitask(dataset=multi_dataset)

        # Train the identifiable linear model
        logger.info("Training the identifiable linear model")
        observations, targets = self.multitask_model.get_latents(multi_dataset, dataloader_cls=(GraphDataLoader if self.use_gnn else None))
        observations = standardize_data(observations, device=self.device)[0]
        with fsspec.open(f"{self.output_path}/data/multitask_model_latents_latent_dim:{self.latent_dim}.pt", "wb") as f:
            torch.save(observations, f)
        with fsspec.open(f"{self.output_path}/data/multitask_model_targets_latent_dim:{self.latent_dim}.pt", "wb") as f:
            torch.save(targets, f)

        task_id = torch.tensor([i for i in range(self.num_tasks) for _ in range(observations.shape[0])]).long()
        x_task = torch.cat([observations for _ in range(self.num_tasks)], dim=0)
        y_task = torch.cat([self.y[:, i] for i in range(self.num_tasks)], dim=0).unsqueeze(-1)
        lin_model_dataset = TensorDataset(x_task, y_task, task_id)

        results = self._train_linear_model(dataset=lin_model_dataset)
        lin_latents = self.linear_model.get_latents(observations, to_numpy=False)
        with fsspec.open(f"{self.output_path}/data/linear_model_latents_latent_dim:{self.latent_dim}.pt", "wb") as f:
            torch.save(lin_latents, f)

        with fsspec.open(f"{self.output_path}/data/c_params.pt", "wb") as f:
            torch.save(self.linear_model.c_params, f)

        # Save parameters of the models to disk
        with fsspec.open(f"{self.output_path}/model_params/multitask_model_params_latent_dim:{self.latent_dim}.pt", "wb") as f:
            torch.save(self.multitask_model.state_dict(), f)

        with fsspec.open(f"{self.output_path}/model_params/linear_model_params_latent_dim:{self.latent_dim}.pt", "wb") as f:
            torch.save(self.linear_model.state_dict(), f)

        # Save results
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
            debug=True,
            use_ground_truth=False,
            batch_size=self.batch_size,
            eval_interval=200,
            use_scheduler=True,
            run_eval=False,
        )

        results_df['seed'] = self.seed
        results_df['num_tasks'] = self.num_tasks
        results_df['latent_dim'] = self.latent_dim
        results_df['sigma_obs'] = self.sigma_obs
        results_df['observed_dim'] = self.observation_dim

        return results_df

    def _train_multitask(self, dataset):

        self.multitask_model = MultiTaskModel(
            observation_dim=self.observation_dim,
            latent_dim=self.latent_dim,
            num_tasks=self.num_tasks,
            hidden_dim=self.hidden_dim,
            device=self.device,
            use_gnn=self.use_gnn,
        )

        self.multitask_model.train_predictor(
            dataset=dataset,
            num_epochs=self.num_multitask_epochs,
            batch_size=self.batch_size,
            cca_dim=self.latent_dim,
            use_scheduler=False,
            track_mcc=False,
            lr=self.lr,
            run_eval=False,
            use_gnn=self.use_gnn,
        )

    def save_results(self, results):
        save_path = f"{self.output_path}/results.csv"
        results.to_csv(save_path)

    @staticmethod
    def collect_results(output_paths):

        result_dfs = []

        # Load the data across all seeds
        for output_path in output_paths:
            config_path = f"{output_path}/config.yaml"
            with fsspec.open(config_path, "r") as f:
                config = yaml.safe_load(f)

            latent_dim = config['latent_dim']
            method = config.get('method', 'MTLCM')

            lin_latents = glob.glob(f"{output_path}/**/linear_model_latents*.pt", recursive=True)
            multi_latents = glob.glob(f"{output_path}/**/multitask_model_latents*.pt", recursive=True)
            lin_latents = [torch.load(l, map_location=torch.device('cpu')) for l in lin_latents]
            multi_latents = [torch.load(l, map_location=torch.device('cpu')) for l in multi_latents]

            # Compute the weak MCC between all pairs of multitask model latents
            multi_weak_mcc = []
            multi_strong_mcc = []
            lin_strong_mcc = []
            for h1, h2 in itertools.combinations(range(len(multi_latents)), 2):
                weak_mcc, strong_mcc = cal_weak_strong_mcc(multi_latents[h1], multi_latents[h2])
                multi_weak_mcc.append(weak_mcc)
                multi_strong_mcc.append(strong_mcc)

            # Compute the strong MCC between all pairs of linear model latents
            for h1, h2 in itertools.combinations(range(len(lin_latents)), 2):
                mcc, _ = cal_mcc(lin_latents[h1], lin_latents[h2])
                lin_strong_mcc.append(mcc)

            result_dfs.append(pd.DataFrame({
                "MTRN (weak)": multi_weak_mcc,
                "multi_strong_mcc": multi_strong_mcc,
                "MTLCM": lin_strong_mcc if len(lin_strong_mcc) > 0 else [np.nan] * len(multi_weak_mcc),
                "latent_dim": latent_dim,
                "method": method,
            }))

        results = pd.concat(result_dfs, ignore_index=True)
        return results

    @staticmethod
    def plot_results(results_dir, save_path, fig_name=None, save_results=False):
        """
        This method takes in the root output directory for the set of ablation results and plots both the weak MCC after
         training and MCC obtained after training the linear model. The plot is saved to disk.

        Args:
            results_dir: str
                Path to the root directory of the ablation results.
            save_path: str
                Path to save the plots.
            fig_name: str
                Name of the figure to save.

        Returns:

        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get the results from the ablation experiment as all results.csv files and concatenate them.
        results = QM9Experiment.collect_results(results_dir)

        if save_results:
            results.to_csv(f"{save_path}/results.csv")

        # Set the colours of the lines to black and red with opacity
        palette = sns.color_palette(["black", "red"], desat=0.5)

        # Plot the results as subplots
        sns.set_style("whitegrid")

        results_columns = ["MTRN (weak)", "MTLCM"]

        for i, col in enumerate(results_columns):
            sns.pointplot(x='latent_dim', y=col, data=results, capsize=0.1, join=False, errwidth=3.0, color=palette[i],
                          label=col)

        # Get figure
        fig = plt.gcf()

        # Set the axis labels
        plt.xlabel("Latent Dimension")
        plt.ylabel("MCC")

        # Set legend outside the plot
        plt.legend(title="Method")

        if fig_name is None:
            fig_name = "qm9_mcc"

        # Save the plot as a pdf, png and svg
        fig.savefig(f"{save_path}/{fig_name}.pdf")
        fig.savefig(f"{save_path}/{fig_name}.png")
        fig.savefig(f"{save_path}/{fig_name}.svg")





