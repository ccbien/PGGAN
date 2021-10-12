from icecream import ic

import torch
from torch import nn


def initialize_layer(layer):
    if layer.weight is not None:
        nn.init.kaiming_normal_(layer.weight, a=0.2)
    if layer.bias is not None:
        layer.bias.data.fill_(0)


class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()

    def forward(self, x1, x2, alpha):
        return (1.0 - alpha) * x1 + alpha * x2


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        mean = torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-8 # dim: channel
        return x / torch.sqrt(mean)