# This file is part of Cytomancer.
#
# Cytomancer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Cytomancer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MyProject. If not, see <http://www.gnu.org/licenses/>.
#
# Portions of this code are based on the original project Pytorch-UNet,
# developed by Milesi Alexandre. Pytorch-UNet is available at:
# <https://github.com/milesial/Pytorch-UNet>
#
# Some modifications made by Jacob Waksmacki, 2025
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        kernel_size: int = 3,
        bilinear: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, kernel_size)
        self.down1 = EncoderBlock(64, 128, kernel_size)
        self.down2 = EncoderBlock(128, 256, kernel_size)
        self.down3 = EncoderBlock(256, 512, kernel_size)
        factor = 2 if bilinear else 1
        self.down4 = EncoderBlock(512, 1024 // factor, kernel_size)

        self.up1 = DecoderBlock(1024, 512 // factor, kernel_size, bilinear)
        self.up2 = DecoderBlock(512, 256 // factor, kernel_size, bilinear)
        self.up3 = DecoderBlock(256, 128 // factor, kernel_size, bilinear)
        self.up4 = DecoderBlock(128, 64, kernel_size, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class DoubleConv(nn.Module):
    """Unet double conv"""

    double_conv: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        kernel_size: int = 3,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Unet encoder with maxpool then double conv"""

    maxpool_conv: nn.Sequential

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    """Unet decoder with double conv then upsample"""

    up: nn.Upsample | nn.ConvTranspose2d
    conv: DoubleConv

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bilinear: bool,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, kernel_size
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Unet out conv"""

    conv: nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
