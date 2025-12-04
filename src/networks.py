import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

class BaseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, activ='relu', norm='instance'):
        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                         stride, padding=padding, bias=False)

        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_ch)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_ch)
        else:
            self.norm = nn.Identity()

        if activ == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()



    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, dim,dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1, padding=0, dilation=dilation),
            nn.InstanceNorm2d((dim)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim),

        )

    def forward(self, x):
        return x + self.block(x)

# ==========================================
#  Stage 1: G1 (Edge Generator) & D1
# ==========================================

class EdgeGenerator(nn.Module):
    """
        Input: RGB Image (3) + Mask (1) = 4 channels (Masked Image)
        Output: Edge Map (1)
    """
    def __init__(self, in_ch=4, n_res_blocks=8):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            BaseBlock(in_ch, 64, kernel=7, stride=1, padding=0, activ='relu'),
            BaseBlock(64, 128,stride=2, activ='relu'),
            BaseBlock(128, 256,stride=2, activ='relu')
        )
        blocks = []
        for _ in range(n_res_blocks):
            blocks.append(ResBlock(256,2))
        self.middle = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BaseBlock(256, 128, kernel=3, stride=1, padding=1, activ='relu'),
            nn.Upsample(scale_factor=2),
            BaseBlock(128, 64, kernel=3, stride=1, padding=1, activ='relu'),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

class EdgeDiscriminator(nn.Module):
    """
        Input: Edge Map (1)
        Output: Patch Real/Fake Prediction
    """
    def __init__(self, in_ch=1):
        super(EdgeDiscriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.model(x)