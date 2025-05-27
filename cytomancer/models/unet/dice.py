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
