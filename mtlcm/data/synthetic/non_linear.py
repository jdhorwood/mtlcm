import torch
from torch.utils.data import Dataset
from mtlcm.utils.data.causal_synthetic import gen_data


class NonLinearDataset(Dataset):
    def __init__(
        self,
        decoder,
        observation_dim,
        latent_dim,
        sigma_s,
        num_causal,
        num_tasks,
        device,
        sample_gammas=True,
        num_support_points=50,
        standardize_features=False,
    ):

        self.decoder = decoder
        self.standardize_features = standardize_features
        self.num_tasks = num_tasks
        self.num_causal = num_causal
        self.num_points_per_task = num_support_points
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.device = device

        x_support, y_support, _, _, self.causal_index, self.gamma_coeffs, _ = gen_data(
            num_tasks=num_tasks,
            num_support_points=num_support_points,
            spurious_noise=sigma_s,
            num_features=latent_dim,
            sigma_range_support=[2, 3],
            num_causal=num_causal,
            target_noise="sigmas",
            flip_spurious=False,
            causal_noise=1,
            sample_gammas=sample_gammas,
            sample_weights=True,
            standardize_features=self.standardize_features,
        )

        self.x_data = x_support.view(-1, self.latent_dim)
        self.y_data = y_support.flatten().unsqueeze(-1)

        # Generate observations
        with torch.no_grad():
            self.obs_data = self.decoder(self.x_data)

        self.causal_index = torch.as_tensor(self.causal_index)

        assert (
            len(self.gamma_coeffs)
            == len(self.causal_index)
            == len(x_support)
            == self.causal_index.shape[0]
            == self.num_tasks
        )

        super().__init__()

    def _get_task_from_index(self, idx):
        return idx // self.num_points_per_task

    def __getitem__(self, idx):

        task_idx = self._get_task_from_index(idx)
        x_data = self.obs_data[idx]
        y_data = self.y_data[idx]

        return x_data, y_data, task_idx

    def __len__(self):
        """
        Returns the length as the number of tasks in the dataset times the number of examples per task.

        Returns: int
        """

        return self.num_tasks * self.num_points_per_task
