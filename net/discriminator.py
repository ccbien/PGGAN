import torch
from torch import nn
from .metrics import PixelNorm, WeightedSum, initialize_layer


class FromRGB(nn.Module):
    def __init__(self, out_channels):
        super(FromRGB, self).__init__()
        self.main = nn.Conv2d(3, out_channels, kernel_size=1, stride=1)
        initialize_layer(self.main)

    def forward(self, x):
        return self.main(x)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        mean_squared_diff = torch.mean(torch.square(x - mean), dim=0, keepdim=True) + 1e-8
        std = torch.sqrt(mean_squared_diff)
        mean_pix = torch.mean(std).view(1, 1, 1, 1).tile(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat((x, mean_pix), dim=1)


class D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D_block, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), 
            PixelNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ]
        initialize_layer(layers[0])
        initialize_layer(layers[3])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self,
        channel_dict={
            4:512, 8:512, 16:512, 32:512, 64:256,
            128:128, 256:64, 512:32, 1024:16            
        },
        last_channels=512,
        device='cuda'
    ):
        super(Discriminator, self).__init__()
        self.channel_dict = channel_dict
        self.last_channels = last_channels
        self.device = device
        self.input_resolution = 4
        self.has_fadein = False
        self.weighted_alpha = 0
        
        layers = [
            MinibatchStddev(),
            nn.Conv2d(channel_dict[4] + 1, channel_dict[4], kernel_size=3, stride=1, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_dict[4], last_channels, kernel_size=4),
            nn.LeakyReLU()
        ]
        initialize_layer(layers[1])
        initialize_layer(layers[4])

        self.from_rgb = FromRGB(channel_dict[4]).to(self.device)
        self.main = nn.Sequential(*layers).to(self.device)
        self.fc = nn.Linear(last_channels, 1).to(self.device)
        initialize_layer(self.fc)
        self.from_rgb_fadein = None
        self.weighted_sum = WeightedSum().to(self.device)
        self.downsample = None
        self.first_block = None


    def forward(self, image):
        if self.has_fadein:
            x1 = self.downsample(image)
            x1 = self.from_rgb_fadein(x1)
            x2 = self.from_rgb(image)
            x2 = self.first_block(x2)
            x2 = self.downsample(x2)
            x = self.weighted_sum(x1, x2, self.weighted_alpha)
        else:
            x = self.from_rgb(image)
        x = self.main(x).view(-1, self.last_channels)
        x = self.fc(x)
        return x

    
    def add_to_main(self, layer):
        self.main = nn.Sequential(layer, *self.main)


    def remove_fadein(self):
        if not self.has_fadein:
            return
        self.has_fadein = False
        self.add_to_main(self.downsample)
        self.add_to_main(self.first_block)
        self.downsample = None
        self.first_block = None
        self.from_rgb_fadein = None


    def add_block(self, fadein=False):
        if self.has_fadein:
            self.remove_fadein()
        in_channels = self.channel_dict[self.input_resolution * 2]
        out_channels = self.channel_dict[self.input_resolution]
        if fadein:
            self.has_fadein = True
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2).to(self.device)
            self.first_block = D_block(in_channels, out_channels).to(self.device)
            self.from_rgb_fadein = self.from_rgb
            self.weighted_alpha = 0
        else:
            self.add_to_main(nn.AvgPool2d(kernel_size=2, stride=2))
            self.add_to_main(D_block(in_channels, out_channels))

        self.from_rgb = FromRGB(in_channels).to(self.device)
        self.input_resolution *= 2


    def set_weighted_alpha(self, alpha):
        self.weighted_alpha = alpha