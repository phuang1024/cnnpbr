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
                nn.BatchNorm2d(out_ch),
                nn.ELU(),
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

    Input   preconv0 -- -- -- -- -- -- -- -- conv3 ---> regression
      |                                        |
    1/2 res preconv1 -- -- -- -- -- - conv2, upsamp
      |                                 |      |
    1/4 res preconv2 -- -- --  conv1, upsamp --
      |                          |      |      |
    1/8 res preconv3 -- conv0, upsamp -- -- -- -
      |
    ...
    """

    def __init__(self):
        super().__init__()

        self.num_channels = []   # Number of input and output channels of conv_n.
        for layer in range(NET_LAYERS):
            if layer == 0:
                self.num_channels.append(NET_PRECONV_CH)
            else:
                num = NET_PRECONV_CH + sum(self.num_channels)
                self.num_channels.append(num)

        for layer in range(NET_LAYERS):
            preconv = NxConv(3, NET_PRECONV_CH, NET_PRECONV_LAYERS, 3)
            self.add_module(f"preconv{layer}", preconv)

            ch = self.num_channels[layer]
            conv = NxConv(ch, ch, NET_CONV_LAYERS, 3)
            self.add_module(f"conv{layer}", conv)

        self.regression = nn.Conv2d(self.num_channels[-1], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Resize to 1/2, 1/4, 1/8, ...
        inputs = [x]
        for i in range(NET_LAYERS-1):
            img = inputs[-1]
            img = nn.functional.interpolate(img, scale_factor=0.5, mode="area")
            inputs.append(img)

        # Preconv layers.
        for i in range(NET_LAYERS):
            inputs[i] = self._modules[f"preconv{i}"](inputs[i])
        inputs = list(reversed(inputs))

        # Conv layers.
        outputs = []
        for i in range(NET_LAYERS):
            # Upscale outputs to current size.
            for j in range(len(outputs)):
                outputs[j] = nn.functional.interpolate(outputs[j], scale_factor=2, mode="bilinear")

            # Concat previous outputs to current input.
            x = torch.cat([*outputs, inputs[i]], dim=1)
            x = self._modules[f"conv{i}"](x)
            outputs.append(x)

        # Regression.
        x = self.regression(x)
        x = self.sigmoid(x)

        return x
