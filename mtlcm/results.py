import fsspec
import yaml
import torch
import itertools
import pandas as pd
import glob
import numpy as np
from mtlcm.utils.data.lin_transform import cal_weak_strong_mcc, cal_mcc
from typer import Typer

app = Typer()

@app.command()
def collect_results(output_path: str):

    config_path = f"{output_path}/config.yaml"
    with fsspec.open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load the data across all seeds
    latent_dim = config["latent_dim"]
    method = config.get("method", "MTLCM")

    lin_latents = glob.glob(
        f"{output_path}/**/linear_model_latents*.pt", recursive=True
    )
    multi_latents = glob.glob(
        f"{output_path}/**/multitask_model_latents*.pt", recursive=True
    )
    lin_latents = [
        torch.load(l, map_location=torch.device("cpu")) for l in lin_latents
    ]
    multi_latents = [
        torch.load(l, map_location=torch.device("cpu")) for l in multi_latents
    ]

    # Compute the weak MCC between all pairs of multitask model latents
    multi_weak_mcc = []
    lin_strong_mcc = []
    for h1, h2 in itertools.combinations(range(len(multi_latents)), 2):
        weak_mcc, _ = cal_weak_strong_mcc(
            multi_latents[h1], multi_latents[h2]
        )
        multi_weak_mcc.append(weak_mcc)

    # Compute the strong MCC between all pairs of linear model latents
    for h1, h2 in itertools.combinations(range(len(lin_latents)), 2):
        mcc, _ = cal_mcc(lin_latents[h1], lin_latents[h2])
        lin_strong_mcc.append(mcc)

    results = pd.DataFrame(
            {
                "MTRN (weak)": multi_weak_mcc,
                "MTLCM": (
                    lin_strong_mcc
                    if len(lin_strong_mcc) > 0
                    else [np.nan] * len(multi_weak_mcc)
                ),
                "latent_dim": latent_dim,
                "method": method,
            }
        )
    
    results.to_csv(f"{output_path}/results.csv", index=False)

if __name__ == "__main__":
    app()