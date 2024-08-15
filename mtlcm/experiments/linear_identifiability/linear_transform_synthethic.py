import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import datamol as dm
from itertools import product
from mtlcm.data.synthetic.linear import LinearDataset
from mtlcm.utils.data.generics import seed_everything
from mtlcm.models.linear_task_model import TaskLinearModel

class LinearTransformExperiment:
    def __init__(self, output_path, latent_dim, num_seeds=1, matrix_types=None, sigma_obs=0.01, sigma_s=0.1,
                 num_causal=2, num_tasks=100, num_points_per_task=50, num_epochs=6000,
                 batch_size=20, device=None, n_jobs=1, standardize_features=True, run_no_causal=False,
                 ground_truth_vals=None, seed=None):

        self.ground_truth_vals = [True, False] if ground_truth_vals is None else ground_truth_vals
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
        self.run_no_causal = run_no_causal

    def run(self):
        """
        Each experiment corresponds to comparing the model in the given matrix setting for a given seed across
        four versions:

            1- No ground truth causal indices or gamma variables
            2- Ground truth causal indices and gamma variables
            3- All variables causal --> leads to non-identifiable result
            4- All variables spurious --> leads to non-identifiable result

        We then plot the MCC and likelihood during training for each model version within each setting.

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

    def setup(self, seed, matrix_type, save_data=True, num_causal_override=None):
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
            num_causal=num_causal_override if num_causal_override is not None else self.num_causal,
            standardize_features=self.standardize_features,
            device=self.device,
        )

        # Save the data
        if save_data and num_causal_override is None:
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

        full_results = []
        seed_everything(seed)
        dataset = self.setup(seed, matrix_type)

        for use_ground_truth in self.ground_truth_vals:
            r = self._train_model(
                dataset=dataset,
                use_ground_truth=use_ground_truth,
                device=self.device,
            )

            # annotate the results with settings
            r["run_type"] = "ground_truth" if use_ground_truth else "no_ground_truth"
            r["matrix_type"] = matrix_type
            r["seed"] = seed
            r["latent_dim"] = self.latent_dim
            r["num_causal"] = self.num_causal
            full_results.append(r)

        if self.run_no_causal:
            for causal_override in (0, self.latent_dim):
                # Here we regenerate the data to illustrate how the model is non-identifiable when all latents are causal
                # or all latents are spurious

                new_data = self.setup(seed, matrix_type, num_causal_override=causal_override)

                r = self._train_model(
                    dataset=new_data,
                    use_ground_truth=use_ground_truth,
                    device=self.device,
                )

                # annotate the results with settings
                r["no_structure_setting"] = "all_causal" if causal_override == self.latent_dim else "all_spurious"
                r["matrix_type"] = matrix_type
                r["seed"] = seed
                full_results.append(r)

        return pd.concat(full_results)

    def _train_model(
        self,
        dataset,
        use_ground_truth,
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
            debug=False,
            use_ground_truth=use_ground_truth,
            batch_size=self.batch_size,
            use_scheduler=True,
        )

        return run_results

    @staticmethod
    def plot_results(results_path, save_path, fig_name=None, cutoff=None):
        """
        Plot the results of the experiment.

        Args:
            cutoff: int
                Specifies the cutoff for the last epoch to plot.
            results_path: str
                Path to the results of the experiment.
            save_path: str
                Path to save the plots.
            fig_name: str
                Name of the figure to save.

        Returns:

        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results = pd.read_csv(results_path)

        # Generate 3 subplots for each matrix type and plot the MCC versus epoch for each ground truth setting
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=2.0)
        # Change the colours of the lines
        sns.set_palette("colorblind")

        results = results.drop(
            labels=[
                l
                for l in results.columns
                if l
                not in (
                    "epoch",
                    "mcc_latents_mean",
                    "run_type",
                    "seed",
                    "matrix_type",
                )
            ],
            axis="columns",
        )

        results = results.rename(columns={'matrix_type': 'Matrix type'})
        # rename the run_type values to remove underscores and capitalize
        results["run_type"] = results["run_type"].apply(lambda x: x.replace("_", " ").capitalize())

        if cutoff is not None:
            results = results[results["epoch"] <= cutoff]

        g = sns.FacetGrid(
            results, col='Matrix type', hue="run_type", height=5, aspect=1.5
        )

        g.map(sns.lineplot, "epoch", "mcc_latents_mean")
        g.set_axis_labels("Epoch", "MCC")

        # Get the axes of the rightmost subplot
        rightmost_ax = g.axes.flatten()[-1]

        # Set the axix ticks size to be larger
        for ax in g.axes.flatten():
            ax.tick_params(axis='both', which='major', labelsize=16)

        # Add the legend inside the rightmost subplot
        legend_ax = plt.subplot(rightmost_ax.get_gridspec()[:, -1])
        legend_ax.legend()

        # Adjust the layout to accommodate the legend
        plt.subplots_adjust(right=0.85)  # Increase the right margin to make space for the legend

        if fig_name is None:
            fig_name = "synthetic_linear_exp"

        # Save figure as pdf, svg and png
        g.savefig(os.path.join(save_path, f"{fig_name}.pdf"))
        g.savefig(os.path.join(save_path, f"{fig_name}.png"))
        g.savefig(os.path.join(save_path, f"{fig_name}.svg"))

    @staticmethod
    def plot_ablation_results(results_dir, save_path, fig_name=None):
        """
        This method takes in the root output directory for the set of ablation results and plots a pointplot of the MCC
        across seeds relative to the latent dimension.

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
        results_paths = glob.glob(os.path.join(results_dir, "**", "results.csv"), recursive=True)

        dfs = []
        for p in results_paths:
            df = pd.read_csv(p)
            if 'num_causal' in df.columns:
                df.rename(columns={'num_causal': 'Number of causal features'}, inplace=True)
            dfs.append(df)

        results_df = pd.concat(dfs, axis=0)

        # Keep only the results for the last epoch
        if "epoch" in results_df.columns:
            results_df = results_df[results_df["epoch"] == results_df["epoch"].max()]

        # Plot the results as a pointplot of the MCC across seeds for each
        # latent dimension with the standard deviation as the error
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=2.0)

        # Set the colours of the lines to black and red with opacity
        sns.set_palette(sns.color_palette(["black", "red"], desat=0.5))

        g = sns.catplot(
            data=results_df,
            x="latent_dim",
            y="mcc_latents_mean",
            hue="Number of causal features",
            kind="point",
            height=5,
            aspect=1.5,
            join=False,
            capsize=0.2,
            legend_out=False,
        )

        # Add grid lines on both axes
        g.ax.xaxis.grid(True)
        g.ax.yaxis.grid(True)
        g.set_axis_labels("Latent Dimension", "MCC")

        g.ax.tick_params(axis='both', which='major', labelsize=16)

        if fig_name is None:
            fig_name = "synthetic_linear_ablation"

        os.makedirs(os.path.join(results_dir, "processed_results"), exist_ok=True)
        results_df['method'] = 'Linear Causal'
        results_df.to_csv(os.path.join(results_dir, "processed_results/results.csv"))

        # Save figure as pdf, svg and png
        g.savefig(os.path.join(save_path, f"{fig_name}.pdf"))
        g.savefig(os.path.join(save_path, f"{fig_name}.png"))
        g.savefig(os.path.join(save_path, f"{fig_name}.svg"))

