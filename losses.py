import torch
from torch import nn


def wasserstein_loss(y, true):
    return torch.mean(y * true)