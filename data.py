import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


def sample_latent(N, size):
    z = torch.randn(N, size)
    return nn.functional.normalize(z)


def get_dataloader(root, resolution, batch_size):
    dts = ImageFolder(
        root = root,
        transform = T.Compose([
            T.Resize(resolution),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    return DataLoader(dts, batch_size=batch_size, shuffle=True, drop_last=True)
