# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_rate=32, scale_ratio=0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_rate (int) - how many filters to add each layer (`k` in paper). Default: 32.
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels + 0 * growth_rate, growth_rate, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels + 1 * growth_rate, growth_rate, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 3, padding=1, bias=True),
                                    nn.LeakyReLU())

        self.scale_ratio = scale_ratio

    def forward(self, inputs):
        layer1 = self.layer1(inputs)
        layer2 = self.layer2(torch.cat((inputs, layer1), 1))
        layer3 = self.layer3(torch.cat((inputs, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((inputs, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((inputs, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.scale_ratio) + inputs


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_rate=32, scale_ratio=0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_rate (int) - how many filters to add each layer (`k` in paper). Default: 32.
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(in_channels, growth_rate)
        self.layer2 = ResidualDenseBlock(in_channels, growth_rate)
        self.layer3 = ResidualDenseBlock(in_channels, growth_rate)
        self.scale_ratio = scale_ratio

    def forward(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.scale_ratio) + inputs


def upsample_block(in_channels, scale_factor=2):
    r"""Realize the function of image magnification.

    Args:
        in_channels (int): Number of channels in the input image.
        scale_factor (int): Image magnification factor.

    Returns:
        nn.Sequential()

    """
    block = []
    for _ in range(scale_factor // 2):
        block += [
            nn.Conv2d(in_channels, in_channels * (2 ** 2), 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]

    return nn.Sequential(*block)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_init_features=64, growth_rate=32, n_basic_block=23,
                 scale_factor=4):
        r"""

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            num_init_features (int): The number of filters to learn in the first convolution layer. Default: 64.
            growth_rate (int): How many filters to add each layer (`k` in paper). Default: 32.
            n_basic_block (int): How many layers in each residual block. Default: 23.
            scale_factor (int): Image magnification factor. Default: 4.
        """
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, num_init_features, 3), nn.ReLU())

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(num_init_features, growth_rate)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(num_init_features, num_init_features, 3), nn.ReLU())
        self.upsample = upsample_block(in_channels=num_init_features, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(num_init_features, num_init_features, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(num_init_features, out_channels, 3), nn.ReLU())

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        out = self.basic_block(conv1)
        out = self.conv2(out)
        out = self.upsample(out + conv1)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class Discriminator(nn.Module):
    r"""Initializing discriminator model"""

    def __init__(self, num_conv_block=4):
        r"""

        Args:
            num_conv_block (int): Number of overlapping residual blocks. Default: 4.
        """
        super(Discriminator, self).__init__()

        block = []

        # Define the initial number of channels
        in_channels = 3
        out_channels = 64

        for _ in range(num_conv_block):
            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3),
                      nn.LeakyReLU(),
                      nn.BatchNorm2d(out_channels)]
            in_channels = out_channels

            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3, 2),
                      nn.LeakyReLU()]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.Conv2d(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(out_channels, out_channels, 3)]

        self.features = nn.Sequential(*block)

        self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

        self.classifier = nn.Sequential(
            nn.Linear(8192, 100),
            nn.Linear(100, 1)
        )

    def forward(self, inputs):
        out = self.features(inputs)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
