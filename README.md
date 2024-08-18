# Overview

This package provides the implementations for the results in the paper:
*Leveraging Task Structures for Improved Identifiability in Neural Network Representations*.

## Installation

1. Create a conda environment with python version 3.9.
    ```bash
        conda create -n mtlcm python=3.9
    ```
2. Install poetry, which we use for dependency management.
3. Activate the conda environment.
3. Within the project's directory, run:

    ```bash
    poetry install
    ```
4. If using cuda, install the cuda version of dgl via pip manually by running (replace cuxxx with your cuda version)
    ```
    poetry remove dgl; pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
    ```
