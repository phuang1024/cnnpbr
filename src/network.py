import torch
import torchvision
from torch import nn

from constants import *


class NxConv(nn.Module):
    """
    Many convolution, batchnorm, leakyrelu layers.
    """

    def __init__(self, in_ch, out_ch, layer_count, kernel_size):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(layer_count):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch, momentum=NET_BN_MOMENTUM),
                nn.ELU(alpha=NET_ELU_ALPHA)
            ))
            in_ch = out_ch

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Network(nn.Module):
    """
    Generates displacement map from color map.
    Many levels of NxConv, MaxPool which will be put together to form the final image.

    Input   preconv0 ----------------------- conv3 ---> regression
      |                                       |
    1/2 res preconv1 ---------------- conv2, upsamp2
      |                                |
    1/4 res preconv2 --------- conv1, upsamp1
      |                         |
    1/8 res preconv3 -- conv0, upsamp0
      |
    ...
    """

    def __init__(self):
        super().__init__()

        for layer in range(NET_LAYERS):
            preconv = NxConv(3, NET_CONV_CH, NET_CONV_LAYERS, NET_CONV_KERNEL)
            self.add_module(f"preconv{layer}", preconv)

            channels = NET_CONV_CH * (layer+1)
            conv = NxConv(channels, channels, NET_CONV_LAYERS, NET_CONV_KERNEL)
            self.add_module(f"conv{layer}", conv)

            if layer < NET_LAYERS - 1:
                upsamp = nn.Upsample(scale_factor=2, mode="bilinear")
                self.add_module(f"upsamp{layer}", upsamp)

        self.regression = nn.Conv2d(NET_CONV_CH * NET_LAYERS, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inputs = [x]
        for i in range(NET_LAYERS-1):
            img = inputs[-1]
            img = torchvision.transforms.functional.resize(img, (img.shape[2]//2, img.shape[3]//2))
            inputs.append(img)

        for i in range(NET_LAYERS):
            inputs[i] = self._modules[f"preconv{i}"](inputs[i])
        inputs = list(reversed(inputs))

        x = inputs[0]
        for i in range(NET_LAYERS):
            if i > 0:
                x = torch.cat([x, inputs[i]], dim=1)
            x = self._modules[f"conv{i}"](x)
            if i < NET_LAYERS - 1:
                x = self._modules[f"upsamp{i}"](x)

        x = self.regression(x)
        x = self.sigmoid(x)
        return x
