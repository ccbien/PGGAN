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
    def __init__(self, delta_p=1e-5, target_rt=0.6):
        super(ADA_rt, self).__init__()
        self.delta_p = delta_p
        self.target_rt = target_rt
        self.p = 0
        self.rt = 0
        self.count = 0
        self.aug = None
        self.set_augmentation()

    def reset(self):
        self.rt = 0
        self.count = 0

    def update_rt(self, y):
        assert len(y.shape) == 2 and y.shape[1] == 1
        sum_sign = float(y.sign().sum())
        n = y.shape[0]
        self.rt = (self.rt * self.count + sum_sign) / (self.count + n)
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
            K.RandomHorizontalFlip(p=p), # x-flip
            K.RandomRotation(p=p, degrees=(90, 90)), # 90 degrees rotation
            K.RandomAffine(p=p, degrees=0, translate=(0.2, 0.2), padding_mode=2), # integer translation
            K.RandomAffine(p=p, degrees=0, scale=(1.1, 1.4), padding_mode=2), # isotropic scaling
            K.RandomAffine(p=p, degrees=(0, 360), padding_mode=2), # arbitrary rotation
            K.ColorJitter(p=p, brightness=(0.7, 1.3)),
            K.ColorJitter(p=p, hue=(-0.5, 0.5)),
            K.ColorJitter(p=p, saturation=(0.5, 2)),
            same_on_batch=False,
            return_transform=False
        )

    def forward(self, x):
        L = x.min()
        R = x.max()
        d = R - L if L < R else 1
        x = (x - L) / d # scale to [0, 1]
        return self.aug(x) * d + L