import os
import typer
import fsspec
import yaml
import time
from mtlcm.experiments.synthetic_multitask.synthetic_multitask import (
    SyntheticMultiTaskExperiment,
)
from mtlcm.experiments.linear_identifiability.linear_transform_synthethic import (
    LinearTransformExperiment,
)
from mtlcm.experiments.superconduct.superconduct_feature_regression import (
    SuperconductFeatureRegressionExperiment,
)
from mtlcm.experiments.qm9.qm9_experiment import QM9Experiment


EXP_CLASSES = {
    "multitask_synthetic": SyntheticMultiTaskExperiment,
    "linear_synthetic": LinearTransformExperiment,
    "superconduct": SuperconductFeatureRegressionExperiment,
    "qm9": QM9Experiment,
}

app = typer.Typer()


@app.command()
def experiment(exp_class: str, config: str):

    start_time = time.time()
    exp_cls = EXP_CLASSES[exp_class]
    with fsspec.open(config, "r") as f:
        config = yaml.safe_load(f)

    output_path = config["output_path"]
    # Save config to output path
    with fsspec.open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    exp = exp_cls(**config)
    exp.run()

    print(f"Experiment took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    app()
