import dgl
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from datamol.utils import fs
from loguru import logger
from mtlcm.utils.data.generics import standardize_data

class GraphDataset:
    """
    Graph dataset for QM9 which uses the DGL format.
    """
    TARGETS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    def __init__(self, save_path=None, preload=True, subset_size: int = None, standardize=False, load_path=None):
        self.load_path = load_path
        self.standardize = standardize
        self.subset_size = subset_size
        self.preload = preload
        self.save_path = save_path
        self._load_data()

    def _load_data(self):

        if self.load_path is not None:
            graphs = pd.read_pickle(fs.join(self.load_path, 'graphs.pkl'))
            labels = pd.read_pickle(fs.join(self.load_path, 'targets.pkl'))
            self.y = torch.as_tensor(labels)
            self.x = graphs
        elif self.preload:
            self.data = dgl.data.QM9EdgeDataset(label_keys=self.TARGETS)
            logger.info("Processing QM9 data")
            graphs, labels = [], []
            subset_index = np.random.choice(a=range(len(self.data)), size=self.subset_size, replace=False) if self.subset_size is not None else range(len(self.data))

            for i in tqdm(subset_index):
                # Doing this will generate the graphs from the source data
                datum = self.data[i]
                graphs.append(datum[0])
                labels.append(datum[1])

            self.x = graphs
            self.y = torch.stack(labels)
            if self.standardize:
                self.y = standardize_data(self.y)[0]
        else:
            self.data = dgl.data.QM9EdgeDataset(label_keys=self.TARGETS)

    def __len__(self):
        return len(self.data) if not self.preload else len(self.x)

    @property
    def num_features(self):
        if self.preload or self.load_path is not None:
            return self.x[0].ndata['attr'].shape[-1]
        return self.data[0][0].ndata['attr'].shape[-1]

    @property
    def num_tasks(self):
        return len(self.TARGETS)

    def __getitem__(self, idx):
        if not self.preload:
            datum = self.data[idx]
            graph = datum[0]
            labels = datum[1]
        else:
            graph = self.x[idx]
            labels = self.y[idx]

        return graph, labels