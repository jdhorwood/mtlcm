import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from mtlcm.models.feature_extractors import EGNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dgl.dataloading.dataloader import GraphDataLoader

from mtlcm.utils.data.lin_transform import cal_weak_strong_mcc


def create_feature_encoder(
    hidden_dim, last_dim, latent_dim, num_hidden_layers, observation_dim, use_gnn=False
):
    if use_gnn:
        return EGNN(
            in_feats=observation_dim,
            h_feats=hidden_dim,
            num_hidden=last_dim if last_dim is not None else latent_dim,
        )
    else:
        if num_hidden_layers == 1:
            feature_encoder = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, last_dim if last_dim is not None else latent_dim),
            )
        elif num_hidden_layers == 2:
            feature_encoder = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, last_dim if last_dim is not None else latent_dim),
            )
        else:
            raise NotImplementedError

    return feature_encoder


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        observation_dim,
        latent_dim,
        num_tasks,
        true_decoder=None,
        hidden_dim=40,
        last_dim=None,
        device=None,
        num_hidden_layers=2,
        use_gnn=False,
    ):
        super(MultiTaskModel, self).__init__()

        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.true_decoder = true_decoder
        self.use_gnn = use_gnn

        self.feature_encoder = create_feature_encoder(
            hidden_dim,
            last_dim,
            latent_dim,
            num_hidden_layers,
            observation_dim,
            use_gnn=use_gnn,
        )

        self.linear_heads = nn.Sequential(
            nn.Linear(last_dim if last_dim is not None else latent_dim, num_tasks),
        )

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.to(self.device)

    def get_representation_from_ground_truth(self, x):
        """
        Returns the representation obtained from passing the observations through the model's encoder.
        These are expected to be linear transformations of the original latent variables.

        Args:
            x: (torch.Tensor) Ground truth data. This will be passed through the true decoder and then the encoder of the model.

        Returns:
            representation: (torch.Tensor) Representation obtained from the encoder of the model.

        """
        with torch.no_grad():
            o = self.true_decoder(x)
            representation = self.feature_encoder(o)
        return representation

    def get_latents(self, dataset, batch_size=1024, to_numpy=True, dataloader_cls=None):
        """
        Returns the latent variables obtained from the encoder of the model over a full dataset.
        Args:
            x: (torch.Tensor) Observations.
            batch_size: (int) Batch size for the dataloader.

        Returns:
            z: (torch.Tensor) Latent variables obtained from the encoder of the model.

        """
        dataloader_cls = dataloader_cls if dataloader_cls is not None else DataLoader
        dataloader = dataloader_cls(dataset, batch_size=batch_size, shuffle=False)
        z = []
        y = []
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                z_batch = self.feature_encoder(x_batch)
                z.append(z_batch)
                y.append(y_batch)
            z = torch.cat(z, dim=0).detach()
            y = torch.cat(y, dim=0).detach()
        if to_numpy:
            z = z.cpu().numpy()
            y = y.cpu().numpy()
        return z, y

    def train_predictor(
        self,
        dataset,
        num_epochs,
        batch_size=32,
        optimizer=None,
        use_scheduler=False,
        lr=1e-3,
        use_gnn=False,
    ):
        dataloader = (
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
            if not use_gnn
            else GraphDataLoader(dataset, shuffle=True, batch_size=batch_size)
        )
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(self.feature_encoder.parameters())
                + list(self.linear_heads.parameters()),
                lr=lr,
            )
        else:
            self.optimizer = optimizer(
                list(self.feature_encoder.parameters())
                + list(self.linear_heads.parameters())
            )  # set any extra parameters as a partial function object (functools)

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

        for _ in tqdm(range(num_epochs)):
            self.train()
            tracked_loss = self._train_epoch(dataloader)

            if use_scheduler:
                scheduler.step(tracked_loss)

    def eval_mcc(self, cca_dim, observations, targets, sample_size=10000):
        # Sample points from the dataset
        sample_size = min(sample_size, observations.shape[0])
        sample_idx = np.random.choice(observations.shape[0], sample_size, replace=False)
        observations = observations[sample_idx]
        targets = targets[sample_idx]

        with torch.no_grad():
            observations = observations.to(self.device)
            r = self.feature_encoder(observations)
            r = r.view(-1, r.shape[-1]).detach().cpu().numpy()
            weak_mcc, strong_mcc = cal_weak_strong_mcc(r, targets, cca_dim=cca_dim)

        return strong_mcc, weak_mcc

    def _train_epoch(
        self,
        dataloader,
    ):
        """
        Traverses through the batches and performs the training for one full epoch.
        """
        self.train()
        epochs_losses = []
        for batch_idx, data_batch in enumerate(dataloader):
            if len(data_batch) == 3:
                inputs, targets, task_idx_batch = data_batch
                task_idx_batch = task_idx_batch.to(self.device)
            else:
                inputs, targets = data_batch
                task_idx_batch = None

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            features = self.feature_encoder(inputs)
            out = self.linear_heads(features)
            if task_idx_batch is not None:
                out = torch.gather(out, index=task_idx_batch.unsqueeze(1), dim=1)

            out = torch.where(torch.isnan(targets), torch.zeros_like(targets), out)
            targets = torch.where(
                torch.isnan(targets), torch.zeros_like(targets), targets
            )
            loss = 0.5 * torch.mean((out - targets) ** 2)
            loss.backward()
            self.optimizer.step()
            epochs_losses.append(loss.item())

        epoch_loss = np.mean(epochs_losses)
        return epoch_loss
