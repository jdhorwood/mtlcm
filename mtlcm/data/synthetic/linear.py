from torch.utils.data import Dataset
from mtlcm.utils.data.lin_transform import generate_synthetic_transform_data


class LinearDataset(Dataset):
    def __init__(
        self,
        observation_dim,
        num_tasks,
        device,
        num_causal=None,
        sigma_obs=0.01,
        sigma_s=0.1,
        o_supportx=None,
        o_supporty=None,
        sample_gammas=True,
        num_support_points=50,
        identity=False,
        orthogonal=False,
        standardize_features=False,
    ):

        self.standardize_features = standardize_features
        self.num_tasks = num_tasks
        self.num_causal = num_causal
        self.num_points_per_task = num_support_points
        self.observation_dim = observation_dim

        # Generate the data if not provided
        if o_supportx is None or o_supporty is None:
            (
                self.gamma_coeffs,
                self.causal_index,
                self.latents,
                self.o_supportx,
                self.o_supporty,
                self.transformation,
            ) = generate_synthetic_transform_data(
                observation_dim=observation_dim,
                sigma_obs=sigma_obs,
                sigma_s=sigma_s,
                num_causal=num_causal,
                num_tasks=num_tasks,
                num_support_points=num_support_points,
                orthogonal=orthogonal,
                identity=identity,
                sample_gammas=sample_gammas,
                device=device,
                standardize_features=self.standardize_features,
            )
        else:
            self.o_supportx = o_supportx
            self.o_supporty = o_supporty
            self.latents = None
            self.transformation = None
            self.causal_index = None
            self.gamma_coeffs = None

        self.x_data = self.o_supportx.view(-1, observation_dim)
        self.y_data = self.o_supporty.flatten().unsqueeze(-1)
        self.latents_flat = (
            self.latents.view(-1, observation_dim) if self.latents is not None else None
        )

        super().__init__()

    @classmethod
    def from_data(
        cls,
        o_supportx,
        o_supporty,
        num_tasks,
        num_support_points,
        device,
        causal_index=None,
        gamma_coeffs=None,
        latents=None,
    ):
        if o_supportx.ndim == 2 and o_supporty.ndim == 2:
            o_supportx = o_supportx.view(num_tasks, num_support_points, -1)
            o_supporty = o_supporty.view(num_tasks, num_support_points, -1)

        observation_dim = o_supportx.shape[-1]

        obj = cls(
            observation_dim=observation_dim,
            num_tasks=num_tasks,
            device=device,
            o_supportx=o_supportx,
            o_supporty=o_supporty,
            num_support_points=num_support_points,
        )

        obj.causal_index = causal_index
        obj.gamma_coeffs = gamma_coeffs
        obj.latents_flat = latents
        obj.latents = (
            latents.view(num_tasks, num_support_points, observation_dim)
            if latents is not None
            else None
        )

        return obj

    def _get_task_from_index(self, idx):
        return idx // self.num_points_per_task

    def __getitem__(self, idx):

        task_idx = self._get_task_from_index(idx)
        x_data = self.x_data[idx]
        y_data = self.y_data[idx]

        return x_data, y_data, task_idx

    def __len__(self):
        """
        Returns the length as the number of tasks in the dataset times the number of examples per task.

        Returns: int
        """

        return self.num_tasks * self.num_points_per_task
