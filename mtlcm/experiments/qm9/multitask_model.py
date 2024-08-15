import torch
import wandb
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader


class MultiTaskModel(nn.Module):
    def __init__(self, observation_dim, hidden_dim, last_dim, num_tasks, device=None):

        super(MultiTaskModel, self).__init__()

        self.observation_dim = observation_dim
        self.last_dim = last_dim
        self.num_tasks = num_tasks

        self.feature_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, last_dim)
        )
        self.linear_heads = nn.Sequential(
            nn.Linear(last_dim, num_tasks),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.to(self.device)


    def train_predictor(
        self,
        x,
        y,
        num_epochs,
        batch_size=32,
        optimizer=None,
    ):     
        
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        wandb.watch(self)
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = []
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                z_batch = self.feature_encoder(x_batch)
                loss = 0.5 * torch.mean((self.linear_heads(z_batch) - y_batch) ** 2)
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            # scheduler.step(tracked_loss)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            wandb.log({"loss": epoch_loss})
            print(epoch+1, loss.item())

        return wandb.run.name, wandb.run.id
    
    def get_features(self, x, run_id, batch_size=1024):
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        z = []
        self.eval()
        with torch.no_grad():
            for x_batch, in dataloader:
                z_batch = self.feature_encoder(x_batch)
                z.append(z_batch)
            z = torch.cat(z, dim=0).detach().cpu().numpy()

        np.save("./experiments/qm9/latents/h{}_{}.npy".format(run_id, self.last_dim), z)


if __name__ == "__main__":
    import argparse
    from sklearn.preprocessing import StandardScaler
    # Process command line arguments for num_epochs, num_tasks, num_causal, observation_dim, fixed_gamma, warmup
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--last_dim", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--run_id", type=int, default=1)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    last_dim = args.last_dim
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    lr = args.lr
    run_id = args.run_id

    x = np.genfromtxt("./experiments/qm9/data/e_all.csv", delimiter=",")
    y = np.genfromtxt("./experiments/qm9/data/y_all.csv", delimiter=",")

    observation_dim = x.shape[-1]
    num_tasks = y.shape[-1]

    wandb.init(project="causal-ml", name="multitask-model", group="qm9")
    model = MultiTaskModel(
        observation_dim=observation_dim, num_tasks=num_tasks, hidden_dim=hidden_dim, last_dim=last_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    x = torch.from_numpy(x).to(model.device).float()
    y = torch.from_numpy(y).to(model.device).float()

    model.train_predictor(
        x=x,
        y=y,
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
    )

    model.get_features(x=x, run_id=run_id)
