import torch
import torch.nn as nn

from constants import *


class ConvDown(nn.Module):
    """
    2x (conv, batchnorm, leakyrelu)
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = kernel_size // 2
        self.one = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(LRELU_ALPHA),
        )

        self.two = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(LRELU_ALPHA),
        )

    def forward(self, x):
        x = self.one(x)
        x = self.two(x)
        return x


class ConvUp(nn.Module):
    """
    2x (conv transpose, batchnorm, leakyrelu)
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = kernel_size // 2
        self.one = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(LRELU_ALPHA),
        )

        self.two = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(LRELU_ALPHA),
        )

    def forward(self, x):
        x = self.one(x)
        x = self.two(x)
        return x


class CNNPBRModel(nn.Module):
    """
    U-net.
    """

    def __init__(self):
        super().__init__()

        # Left, down layers
        for i in range(LAYERS):
            in_channels = 3 if i == 0 else getattr(self, f"down{i-1}").out_channels
            out_channels = 2 ** (i+CHANNELS_EXP)

            conv = ConvDown(in_channels, out_channels, KERNEL_SIZES[i])
            setattr(self, f"down{i}", conv)

            if i != LAYERS-1:
                maxpool = POOLING(2)
                setattr(self, f"pooling{i}", maxpool)

        # Right, up layers
        for i in range(LAYERS-1):
            in_channels = getattr(self, f"down{LAYERS-1}").out_channels if i == 0 else \
                getattr(self, f"up{i-1}").out_channels

            upsamp = nn.Upsample(scale_factor=2, mode="bilinear")
            setattr(self, f"upsamp{i}", upsamp)

            # Concatenates upsamp with corresponding down layer
            real_in_channels = getattr(self, f"down{LAYERS-i-2}").out_channels + in_channels
            out_channels = real_in_channels // 2
            conv = ConvUp(real_in_channels, out_channels, KERNEL_SIZES[-i-2])
            setattr(self, f"up{i}", conv)

        # Regression (output)
        in_channels = getattr(self, f"up{LAYERS-2}").out_channels
        in_channels += 3   # Concatenate with input

        self.reg_nrmr = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.reg_nrmg = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.reg_nrmb = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.reg_disp = nn.Conv2d(in_channels, 1, 1)
        self.reg_rough = nn.Conv2d(in_channels, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x

        # Left, down layers
        lefts = []
        for i in range(LAYERS):
            x = getattr(self, f"down{i}")(x)
            lefts.append(x)
            if i != LAYERS-1:
                x = getattr(self, f"pooling{i}")(x)

        # Right, up layers
        for i in range(LAYERS-1):
            x = getattr(self, f"upsamp{i}")(x)
            x = torch.cat([x, lefts[LAYERS-i-2]], dim=1)
            x = getattr(self, f"up{i}")(x)
        x = torch.cat([x, original], dim=1)

        nrmr = self.reg_nrmr(x)
        nrmg = self.reg_nrmg(x)
        nrmb = self.reg_nrmb(x)
        disp = self.reg_disp(x)
        rough = self.reg_rough(x)
        final = torch.cat([nrmr, nrmg, nrmb, disp, rough], dim=1)

        final = self.sigmoid(final)
        return final
