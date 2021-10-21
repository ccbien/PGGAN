from typing import ForwardRef
from icecream import ic

import torch
from torch import nn
import kornia.augmentation as K


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


class ADA_rt(nn.Module):
    def __init__(self, delta_p=0.1, target_rt=0.6):
        super(ADA_rt, self).__init__()
        self.delta_p = delta_p
        self.target_rt = target_rt
        self.p = 0
        self.rt = 0
        self.count = 0
        self.aug = None
        self.set_augmentation()

    def update_rt(self, y):
        assert len(y.shape) == 2 and y.shape[1] == 1
        sum_sign = float(y.sign().sum())
        n = y.shape[0]
        self.rt = (self.rt * self.count + sum_sign * n) / (self.count + n)
        self.count += n

    def adjust_p(self):
        if self.rt < self.target_rt: # overfitting
            self.p = min(self.p + self.delta_p, 1)
        elif self.rt > self.target_rt:
            self.p = max(self.p - self.delta_p, 0)
        self.set_augmentation()

    def set_augmentation(self):
        p = self.p
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p),
            same_on_batch=False,
            return_transform=False
        )

    def forward(self, x):
        return self.aug(x)