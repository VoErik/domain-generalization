import torch
import torch.nn as nn
from torch import Tensor


class DenseBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            growth_rate,
            num_layers
    ):
        super(DenseBlock, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(self._make_layer((in_channels + i * growth_rate), growth_rate))
        self.block = nn.Sequential(*layers)

    @staticmethod
    def _make_layer(in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.block:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x
