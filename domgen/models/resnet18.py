import torch.nn as nn

from torch import Tensor
from typing import Type
from BasicBlock import BasicBlock

# https://www.geeksforgeeks.org/resnet18-from-scratch-using-pytorch/
# https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/


class ResNet18(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        num_classes: int
    ):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)  # notwendig?
        self.relu = nn.ReLU(inplace=True)  # notwendig?
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers/Blöcke
        self.layer1 = self._make_layer(block, 64, 2, stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, 2, stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, 2, stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, 2, stride=2)  # conv5_x

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
            self,
            block: Type[BasicBlock],
            out_channels: int,
            num_blocks: int,
            stride: int = 1
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
