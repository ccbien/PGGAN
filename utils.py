import os
from icecream import ic
import numpy as np
from skimage.io import imsave
from skimage.transform import rescale
import torch


def log(string, dir, end='\n'):
    path = os.path.join(dir, 'log.txt')
    with open(path, 'a') as f:
        f.write(string + end)
        print(string, end=end)


def get_rgb(x): # Convert torch.Tensor to numpy.uint8 array
    if len(x.shape) == 3:
        img = x.permute(1, 2, 0).cpu().detach().numpy()
    elif len(x.shape) == 4:
        img = x.permute(0, 2, 3, 1).cpu().detach().numpy()
    else:
        raise ValueError('Invalid shape')
    img = (img + 1) * 127.5
    return img.astype('uint8')


def save_images(G, z, path, upscale=None, padding_size=None):
    with torch.no_grad():
        x = G(z)
    if upscale is not None:
        x = torch.nn.Upsample(scale_factor=upscale)(x)
    images = get_rgb(x)
    if padding_size is not None:
        p = padding_size
        images = np.pad(images, pad_width=((0,0), (p, p), (p, p), (0, 0)))
    res = images[0]
    for i in range(1, z.shape[0]):
        res = np.concatenate([res, images[i]], axis=1)
    imsave(path, res)
    

def get_alpha(count):
    """Generate alpha values for face-in layer"""
    for i in range(count):
        yield (i + 1) / (count + 1)