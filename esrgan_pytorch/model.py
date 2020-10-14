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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import vgg19


class Discriminator(nn.Module):
    r"""Discriminator similar to VGG structure"""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # Conv0
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

        self.classifier = nn.Sequential(
            nn.Linear(8192, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, input: Tensor = None) -> Tensor:
        out = self.features(input)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class FeatureExtractorVGG22(nn.Module):
    r"""A loss defined on feature maps representing lower-level features.
    `"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" <https://arxiv.org/pdf/1603.08155.pdf>`_
    """

    def __init__(self, feature_layer: int = 8) -> None:
        """ Constructing characteristic loss function of VGG network. For VGG2.2.

        Args:
            feature_layer (int): How many layers in VGG19. (Default:8).

        Notes:
            features(
              (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
              (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): ReLU(inplace=True)
              (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (6): ReLU(inplace=True)
              (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        super(FeatureExtractorVGG22, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer]).eval()

    def forward(self, input: Tensor = None) -> Tensor:
        out = self.features(input)

        return out


class FeatureExtractorVGG54(nn.Module):
    r"""A loss defined on feature maps representing lower-level features.
    `"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" <https://arxiv.org/pdf/1603.08155.pdf>`_

    A loss defined on feature maps of higher level features from deeper network layers
    with more potential to focus on the content of the images. We refer to this network
    as SRGAN in the following.
    """

    def __init__(self, feature_layer: int = 35) -> None:
        """ Constructing characteristic loss function of VGG network. For VGG5.4.

        Args:
            feature_layer (int): How many layers in VGG19. (Default:35).

        Notes:
            features(
              (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
              (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): ReLU(inplace=True)
              (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (6): ReLU(inplace=True)
              (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (8): ReLU(inplace=True)
              (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (11): ReLU(inplace=True)
              (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (13): ReLU(inplace=True)
              (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (15): ReLU(inplace=True)
              (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (17): ReLU(inplace=True)
              (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (20): ReLU(inplace=True)
              (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (22): ReLU(inplace=True)
              (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (24): ReLU(inplace=True)
              (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (26): ReLU(inplace=True)
              (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (29): ReLU(inplace=True)
              (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (31): ReLU(inplace=True)
              (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (33): ReLU(inplace=True)
              (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

        """
        super(FeatureExtractorVGG54, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer]).eval()

    def forward(self, input: Tensor = None) -> Tensor:
        out = self.features(input)

        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor):
        r""" This is an esrgan model defined by the author himself.

        Args:
            upscale_factor (int): Image magnification factor. (Default: 4).
        """
        super(Generator, self).__init__()
        self.upsample_block_num = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)

        # 23 ResidualInResidualDenseBlock layer
        rrdb_layers = []
        for _ in range(23):
            rrdb_layers += [ResidualInResidualDenseBlock(64, 32)]
        self.residual_residual_dense_blocks = nn.Sequential(*rrdb_layers)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Upsampling layers
        self.upsampling = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, input: Tensor) -> Tensor:
        out1 = self.conv1(input)
        out = self.residual_residual_dense_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        for _ in range(self.upsample_block_num):
            F.leaky_relu(input=self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest")),
                         negative_slope=0.2, inplace=True)
        out = self.conv3(out)
        return out


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, in_channels=64, growth_channels=32, scale_ratio=0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 0 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + 1 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.scale_ratio = scale_ratio

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))
        return conv5 * self.scale_ratio + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels=64, growth_channels=32, scale_ratio=0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RBD1 = ResidualDenseBlock(in_channels, growth_channels)
        self.RBD2 = ResidualDenseBlock(in_channels, growth_channels)
        self.RBD3 = ResidualDenseBlock(in_channels, growth_channels)
        self.scale_ratio = scale_ratio

    def forward(self, input: Tensor) -> Tensor:
        out = self.RBD1(input)
        out = self.RBD2(out)
        out = self.RBD3(out)
        return out * self.scale_ratio + input
