import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Set the seed for torch and numpy.
    Args:
        seed: Random seed.

    Returns:
        None

    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True


def freeze(params):
    if isinstance(params, torch.nn.Parameter):
        params.requires_grad = False
    else:
        for param in params:
            param.requires_grad = False


def unfreeze(params):
    if isinstance(params, torch.nn.Parameter):
        params.requires_grad = True
    else:
        for param in params:
            param.requires_grad = True


def standardize_data(*tensors, device="cpu"):
    return [
        torch.from_numpy(StandardScaler().fit_transform(t)).to(device).float()
        for t in tensors
    ]
