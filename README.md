# Leveraging Task Structures for Improved Identifiability in Neural Network Representations (TMLR 2024)

<div align="center">

[![Paper](https://img.shields.io/badge/paper-arxiv.2306.14861-red)](https://arxiv.org/abs/2306.14861)
[![Paper](https://img.shields.io/badge/TMLR-2024-blue)](https://openreview.net/forum?id=WLcPrq6pu0)

</div>

This package provides the implementations for the results in the paper:
*Leveraging Task Structures for Improved Identifiability in Neural Network Representations* (published at [TMLR 2024](https://openreview.net/forum?id=WLcPrq6pu0)).

## Installation

1. Create a conda environment with python version 3.9.
    ```bash
    conda create -n mtlcm python=3.9
    ```
2. Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer), which we use for dependency management.
3. Activate the conda environment.
    ```bash
    conda activate mtlcm
    ```
4. Within the project's directory, run:
    ```bash
    poetry install
    ```
4. If using cuda, install the cuda version of dgl via pip manually by running (replace cuxxx with your cuda version)
    ```
    poetry remove dgl; pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
    ```

## Running the experiments

The experiments are run via the file `mtlcm/run.py` and configured through yaml files. Below are sample commands for each experiment. Replace the configuration files/entries to try different hyper-parameters.

### Linear synthetic data

```python
python mtlcm/run.py linear_synthetic mtlcm/experiments/linear_identifiability/configs/config.yaml
```

### Non-linear synthetic data

```python
python mtlcm/run.py multitask_synthetic mtlcm/experiments/synthetic_multitask/configs/exp_config.yaml
```

### QM9 Data

```python
python mtlcm/run.py qm9 mtlcm/experiments/qm9/configs/latent_7/config_0.yaml
```

### Superconductivity data

```python
python mtlcm/run.py superconduct mtlcm/experiments/superconduct/configs/test_config.yaml
```

## Results

Results for synthetic data experiments will be immediately available in the exp_outputs/ directory. For real-world data, the above command should be run multiple times for different seeds (e.g. latent_7/config_[1,2,3,4].yaml in the above example for qm9). The results can then be computed by comparing the representations across seeds by running 

```python
python mtlcm/results.py "exp_outputs/qm9/full_config_latent7"
```


