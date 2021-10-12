import os
from icecream import ic
import shutil
from numpy import fix
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils import data
from losses import wasserstein_loss
from data import get_dataloader, sample_latent
from net.generator import Generator
from net.discriminator import Discriminator

from utils import get_alpha, log, save_images


def train_G_on_batch(G, opt_G, D, N):
    opt_G.zero_grad()
    z = sample_latent(N, G.latent_size).to(G.device)
    x = G(z)
    y = D(x)
    loss = wasserstein_loss(y, 1)
    loss.backward()
    opt_G.step()
    return loss.item()


def train_D_on_batch(D, opt_D, G, x, N):
    opt_D.zero_grad()
    y = D(x.to(D.device))
    loss1 = wasserstein_loss(y, 1)
    loss1.backward()
    opt_D.step()

    opt_D.zero_grad()
    z = sample_latent(N, G.latent_size).to(D.device)
    x = G(z)
    y = D(x)
    loss2 = wasserstein_loss(y, -1)
    loss2.backward()
    opt_D.step()

    return (loss1.item() + loss2.item()) / 2


def train_on_epoch(epoch, G, D, opt_G, opt_D, resolution, alphas, config):
    log('----------------------------------------------------------------------', config.dir)
    log('----------------------------------------------------------------------', config.dir)
    log(f'EPOCH {epoch} -- RESOLUTION: {resolution}', config.dir)

    N = config.batch_size_dict[resolution] # batch_size for current resolution
    dataloader = get_dataloader(config.dataroot, resolution, N)
    g_losses, d_losses = [], []

    for it, (x, _) in enumerate(dataloader, 1):
        alpha = next(alphas)
        G.set_weighted_alpha(alpha)
        D.set_weighted_alpha(alpha)

        d_loss = train_D_on_batch(D, opt_D, G, x, N)
        g_loss = train_G_on_batch(G, opt_G, D, N)
        
        log('Epoch=%03d [%06d/%d]: g_loss = %15.4f,         d_loss = %15.4f' % (epoch, it, len(dataloader), g_loss, d_loss), config.dir)
        g_losses.append(g_loss)
        d_losses.append(d_loss)

    g_loss = sum(g_losses) / len(g_losses)
    d_loss = sum(d_losses) / len(d_losses)
    log('Mean:  g_loss=%.4f,  d_loss=%.4f' % (g_loss, d_loss), config.dir)
        

def save_on_epoch(epoch, G, D, fixed_z, resolution, config, save_checkpoint=False):
    if save_checkpoint:
        state = {
            'G': G.state_dict(),
            'D': D.state_dict()
        }
        torch.save(state, os.path.join(config.chkpt_dir, 'ep-%03d.pth' % epoch))
    
    upscale = None if resolution >= 128 else 128 / resolution
    impath = os.path.join(config.demo_dir, 'ep-%03d.jpg' % epoch)
    save_images(G, fixed_z, impath, upscale, padding_size=10)


def train(args):
    config = OmegaConf.load(args.config)
    config.dir = os.path.join('train/', args.run_name)
    config.chkpt_dir = os.path.join(config.dir, 'chkpt/')
    config.demo_dir = os.path.join(config.dir, 'demo/')

    shutil.copyfile(args.config, os.path.join(config.dir, 'config.yaml'))
    os.makedirs(config.chkpt_dir)
    os.makedirs(config.demo_dir)

    G = Generator(
        latent_size=config.latent_size,
        channel_dict=config.channel_dict,
        device=config.device
    )
    D = Discriminator(
        last_channels=config.latent_size,
        channel_dict=config.channel_dict,
        device=config.device,
    )

    opt_G = Adam(G.parameters(), config.lr, (config.b1, config.b2), config.eps)
    opt_D = Adam(D.parameters(), config.lr, (config.b1, config.b2), config.eps)

    fixed_z = sample_latent(4, G.latent_size).to(config.device)
    resolution = 4
    start_epoch = 1
    while True:
        batch_size = config.batch_size_dict[resolution]
        dataloader = get_dataloader(config.dataroot, resolution, batch_size)
        alphas = get_alpha(config.epochs * len(dataloader))
        del dataloader, batch_size

        for epoch in range(start_epoch, start_epoch + config.epochs):
            train_on_epoch(epoch, G, D, opt_G, opt_D, resolution, alphas, config)
            save_on_epoch(
                epoch, G, D, fixed_z, resolution, config,
                save_checkpoint=(epoch % config.checkpoint_frequency == 0)
            )

        if resolution < config.target_resolution:
            resolution *= 2
            G.add_block(fadein=True)
            D.add_block(fadein=True)
            start_epoch += config.epochs
        else:
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--run_name', type=str, required=True, help='Run name')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing folder')
    args = parser.parse_args()

    root = os.path.join('./train/', args.run_name)
    if args.overwrite and os.path.exists(root):
        shutil.rmtree(root)

    try:
        os.makedirs(root)
    except FileExistsError:
        print('Run name alrealy exists.')
        exit()

    train(args)