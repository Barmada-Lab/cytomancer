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
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor):
    if not input.size() == target.size():
        raise ValueError("Input and target must have the same size")
    if not input.dim() == 3:
        raise ValueError("Input must have 3 dimensions")

    sum_dim = (-1, -2, -3)

    intersection = 2 * (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    union = torch.where(union == 0, intersection, union)

    dice = (intersection + 1e-6) / (union + 1e-6)

    return dice.mean()


def dice_loss(input: Tensor, target: Tensor):
    return 1 - dice_coeff(input, target)
