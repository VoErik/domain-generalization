import torch.nn as nn
from torch import Tensor
from typing import Type
from BasicBlock import BasicBlock
from Bottleneck import Bottleneck


class ResNet(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: list, num_classes: int):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[nn.Module], out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            downsample = None
            if stride != 1 or self.in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels * block.expansion

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

    @staticmethod
    def resnet18(num_classes: int):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

    @staticmethod
    def resnet50(num_classes: int):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
