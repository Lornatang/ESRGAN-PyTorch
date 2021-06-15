# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .utils import ResidualInResidualDenseBlock

model_urls = {
    "esrgan16": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/v0.2.0/ESRGAN16_DF2K-a03a643d.pth",
    "esrgan23": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/v0.2.0/ESRGAN23_DF2K-13a67ca9.pth"
}


class Generator(nn.Module):
    def __init__(self, num_rrdb_blocks: int = 16):
        r""" This is an esrgan model defined by the author himself.

        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.

        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).

        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16/23 ResidualInResidualDenseBlock layer.
        trunk = []
        for _ in range(num_rrdb_blocks):
            trunk += [ResidualInResidualDenseBlock(channels=64, growth_channels=32, scale_ratio=0.2)]
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        trunk = self.trunk(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = F.leaky_relu(self.up1(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.up2(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


def _gan(arch, num_residual_block, pretrained, progress) -> Generator:
    model = Generator(num_residual_block)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def esrgan16(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from `<https://arxiv.org/pdf/1809.00219.pdf>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("esrgan16", 16, pretrained, progress)


def esrgan23(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from `<https://arxiv.org/pdf/1809.00219.pdf>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("esrgan23", 23, pretrained, progress)
