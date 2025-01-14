# TODO: implement MixUp method and integrate into training
# proposed in: Zhang et al. 2018: https://arxiv.org/abs/1710.09412
# github repo: https://github.com/facebookresearch/mixup-cifar10/tree/main - look in train function for mixup data and
# mixup criterion
from typing import Tuple

import numpy as np
import torch


def mixup_data(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0,
        device: str = 'cpu'
) -> Tuple:
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
        criterion: torch.nn.Module,
        pred:torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float
) -> float:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
