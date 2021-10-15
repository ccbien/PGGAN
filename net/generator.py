from torch import nn
from .metrics import PixelNorm, WeightedSum, initialize_layer


class ToRGB(nn.Module):
    def __init__(self, in_channels, flag_tanh):
        super(ToRGB, self).__init__()
        self.main = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1)
        initialize_layer(self.main)
        self.tanh = nn.Tanh() if flag_tanh else None

    def forward(self, x):
        x = self.main(x)
        return x if self.tanh is None else self.tanh(x)


class G_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G_block, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            PixelNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ]
        initialize_layer(layers[0])
        initialize_layer(layers[3])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class Generator(nn.Module):
    def __init__(self,
        latent_size=512,
        channel_dict={
            4:512, 8:512, 16:512, 32:512, 64:256,
            128:128, 256:64, 512:32, 1024:16            
        },
        device='cuda',
        flag_tanh=True
    ):
        super(Generator, self).__init__()
        self.device = device
        self.flag_tanh = flag_tanh
        self.latent_size = latent_size
        self.channel_dict = channel_dict
        self.output_resolution = 4
        self.has_fadein = False
        self.weighted_alpha = 0

        layers = [
            nn.ConvTranspose2d(latent_size, channel_dict[4], kernel_size=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_dict[4], channel_dict[4], kernel_size=3, stride=1, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ]
        initialize_layer(layers[0])
        initialize_layer(layers[2])

        self.main = nn.Sequential(*layers).to(self.device)
        self.to_rgb = ToRGB(in_channels=channel_dict[4], flag_tanh=self.flag_tanh).to(self.device)
        self.to_rgb_fadein = None
        self.weighted_sum = WeightedSum().to(self.device)
        self.upsample = None
        self.last_block = None


    def forward(self, z):
        if len(z.shape) != 4:
            z = z.view(-1, self.latent_size, 1, 1)
        x = self.main(z)
        if self.has_fadein:
            x = self.upsample(x)
            x1 = self.to_rgb_fadein(x)
            x2 = self.last_block(x)
            x2 = self.to_rgb(x2)
            x = self.weighted_sum(x1, x2, self.weighted_alpha)
        else:
            x = self.to_rgb(x)
        return x


    def add_to_main(self, layer):
        # self.main.add_module(str(len(self.main)), layer)
        self.main = nn.Sequential(*self.main, layer)


    def remove_fadein(self):
        if not self.has_fadein:
            return
        self.has_fadein = False
        self.add_to_main(self.upsample)
        self.add_to_main(self.last_block)
        self.upsample = None
        self.last_block = None
        self.to_rgb_fadein = None


    def add_block(self, fadein=False):
        if self.has_fadein:
            self.remove_fadein()
        in_channels = self.channel_dict[self.output_resolution]
        out_channels = self.channel_dict[self.output_resolution * 2]
        if fadein:
            self.has_fadein = True
            self.upsample = nn.Upsample(scale_factor=2).to(self.device)
            self.last_block = G_block(in_channels, out_channels).to(self.device)
            self.to_rgb_fadein = self.to_rgb
            self.weighted_alpha = 0
        else:
            self.add_to_main(nn.Upsample(scale_factor=2).to(self.device))
            self.add_to_main(G_block(in_channels, out_channels).to(self.device))

        self.to_rgb = ToRGB(out_channels, flag_tanh=self.flag_tanh).to(self.device)
        self.output_resolution *= 2

    
    def set_weighted_alpha(self, alpha):
        self.weighted_alpha = alpha
