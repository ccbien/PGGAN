import os
import shutil
from time import time
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils import data
from losses import wasserstein_loss
from data import get_dataloader, sample_latent
from net.generator import Generator
from net.discriminator import Discriminator, WeightClipper
from net.metrics import ADA_rt

from utils import get_alpha, log, save_images


def train_G_on_batch(N):
    global args, config, G, D, opt_G, opt_D, ada

    opt_G.zero_grad()
    z = sample_latent(N, G.latent_size).to(G.device)
    x = G(z)
    x_aug = ada(x)
    y = D(x)
    loss = wasserstein_loss(y, -1)
    loss.backward()
    opt_G.step()
    return loss.item()


def train_D_on_batch(x, N):
    global args, config, G, D, opt_G, opt_D, ada

    opt_D.zero_grad()
    x_aug = ada(x.to(D.device))
    y = D(x_aug)
    ada.update_rt(y)
    loss1 = wasserstein_loss(y, -1)
    loss1.backward()
    
    z = sample_latent(N, G.latent_size).to(D.device)
    x = G(z)
    x_aug = ada(x)
    y = D(x)
    loss2 = wasserstein_loss(y, 1)
    loss2.backward()
    opt_D.step()
    D.apply(WeightClipper(0.01))

    return (loss1.item() + loss2.item()) / 2


def train_on_epoch(epoch, resolution, alphas):
    global args, config, G, D, opt_G, opt_D, ada

    log('-' * 70, config.dir)
    log(f'EPOCH {epoch} -- RESOLUTION: {resolution}', config.dir)
    start_time = time()

    N = config.batch_size_dict[resolution] # batch_size for current resolution
    dataloader = get_dataloader(config.dataroot, resolution, N)
    g_losses, d_losses = [], []
    ada.reset()

    for it, (x, _) in enumerate(dataloader, 1):
        if alphas is not None:
            alpha = next(alphas)
            G.set_weighted_alpha(alpha)
            D.set_weighted_alpha(alpha)

        d_loss = train_D_on_batch(x, N)
        g_loss = train_G_on_batch(N)

        if it % 4 == 0:
            ada.adjust_p()
        
        # log('Epoch=%03d [%06d/%d]: g_loss = %15.4f,         d_loss = %15.4f' % (epoch, it, len(dataloader), g_loss, d_loss), config.dir)
        g_losses.append(g_loss)
        d_losses.append(d_loss)

    g_loss = sum(g_losses) / len(g_losses)
    d_loss = sum(d_losses) / len(d_losses)
    log('g_loss = %10.2f         d_loss = %10.2f         p = %7.5f         rt = %7.5f         time = %4ds' % (g_loss, d_loss, ada.p, ada.rt, int(time() - start_time)), config.dir)        


def save_on_epoch(epoch, fixed_z, resolution):
    global args, config, G, D, opt_G, opt_D

    upscale = None if resolution >= 128 else 128 / resolution
    impath = os.path.join(config.demo_dir, 'ep-%03d.jpg' % epoch)
    save_images(G, fixed_z, impath, upscale, padding_size=None)
    
    state = {
        'G': G.state_dict(),
        'D': D.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
        'fixed_z': fixed_z,
        'epoch': epoch
    }
    torch.save(state, os.path.join(config.chkpt_dir, 'last.pth'))


def load_state_dicts():
    global args, config, G, D, opt_G, opt_D
    cp = torch.load(os.path.join(config.chkpt_dir, 'last.pth'))
    G.load_state_dict(cp['G'])
    D.load_state_dict(cp['D'])
    opt_G.load_state_dict(cp['opt_G'])
    opt_D.load_state_dict(cp['opt_D'])


def train(resume=False):
    global args, config, G, D, opt_G, opt_D, ada

    if not resume:
        shutil.copyfile(args.config, os.path.join(config.dir, 'config.yaml'))
        os.makedirs(config.chkpt_dir)
        os.makedirs(config.demo_dir)

    G = Generator(
        latent_size=config.latent_size,
        channel_dict=config.channel_dict,
        device=config.device,
        flag_tanh=config.flag_tanh
    )
    D = Discriminator(
        last_channels=config.latent_size,
        channel_dict=config.channel_dict,
        device=config.device,
    )
    opt_G = Adam(G.parameters(), config.lr, (config.b1, config.b2), config.eps)
    opt_D = Adam(D.parameters(), config.lr, (config.b1, config.b2), config.eps)

    if resume:
        cp = torch.load(os.path.join(config.chkpt_dir, 'last.pth'))
        fixed_z = cp['fixed_z']
        cp_last_epoch = cp['epoch']
    else:
        cp = None
        fixed_z = sample_latent(8, G.latent_size).to(config.device)
        cp_last_epoch = 0

    ada = ADA_rt(delta_p=config.delta_p)
    resolution = 4
    start_epoch = 1


    while True:
        batch_size = config.batch_size_dict[resolution]
        dataloader = get_dataloader(config.dataroot, resolution, batch_size)
        alphas = get_alpha(config.epochs * len(dataloader))
        del dataloader, batch_size

        # Growing phase
        for epoch in range(start_epoch, start_epoch + config.epochs):
            if epoch > cp_last_epoch:
                if epoch == cp_last_epoch + 1 and cp_last_epoch > 0:
                    load_state_dicts()
                train_on_epoch(epoch, resolution, alphas)
                save_on_epoch(epoch, fixed_z, resolution)

        # Stabilization phase
        G.remove_fadein()
        D.remove_fadein()
        start_epoch += config.epochs
        for epoch in range(start_epoch, start_epoch + config.epochs):
            if epoch > cp_last_epoch:
                if epoch == cp_last_epoch + 1 and cp_last_epoch > 0:
                    load_state_dicts()
                train_on_epoch(epoch, resolution, alphas=None)
                save_on_epoch(epoch, fixed_z, resolution)

        if resolution < config.target_resolution:
            resolution *= 2
            G.add_block(fadein=True)
            D.add_block(fadein=True)
            start_epoch += config.epochs
        else:
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Whether to resume training or not')
    parser.add_argument('-r', '--run_name', type=str, required=True, help='Run name')
    parser.add_argument('-c', '--config', type=str, default='', help='Path to configuration file')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing folder')
    args = parser.parse_args()

    root = os.path.join('./train/', args.run_name)
    if args.resume:
        config_path = os.path.join(root, 'config.yaml')
    else:
        config_path = args.config

    config = OmegaConf.load(config_path)
    config.dir = root
    config.chkpt_dir = os.path.join(config.dir, 'chkpt/')
    config.demo_dir = os.path.join(config.dir, 'demo/')
    G = D = opt_G = opt_D = ada = None
    
    if not args.resume:
        if args.overwrite and os.path.exists(root):
            shutil.rmtree(root)

        try:
            os.makedirs(root)
        except FileExistsError:
            print('Run name alrealy exists, consider adding --overwrite flag.')
            exit()

    train(args.resume)