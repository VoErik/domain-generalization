from typing import List

import torch


def denormalize(
        tensor: torch.Tensor,
        mean: List[float],
        std: List[float],
) -> torch.Tensor:
    r"""
    Denormalize a tensor using a mean and standard deviation.
    The denormalization formula for each channel `c` is:

    .. math::
        \text{denormalized\_value}[c] = (\text{normalized\_value}[c] \times \text{std}[c]) + \text{mean}[c]

    :param tensor: The normalized tensor. Shape should be (C, H, W).
    :param mean: List of means for each channel.
    :param std: List of standard deviations for each channel.
    :return: The denormalized tensor.
    """
    # so the mean + std are broadcastable
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    denormalized_tensor = tensor * std + mean
    return denormalized_tensor