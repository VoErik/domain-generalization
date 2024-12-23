import torch.nn as nn
from torch import Tensor
from typing import Type
import torchvision.models as models
from copy import deepcopy


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
        input_channels = x.size(1)
        if input_channels != 3:
            conv_adjust = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, bias=False)
            x = conv_adjust(x)

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


class BasicBlock (nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: 1,
            downsample=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # skip layer
        out = self.relu(out)

        return out


class Bottleneck (nn.Module):
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample=None
    ):
        super(Bottleneck, self).__init__()
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution to process spatial dimensions
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution to expand channels back
        self.conv3 = nn.Conv2d(
            out_channels,
            self.expansion * out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # skip layer
        out = self.relu(out)

        return out


def _resnet(model_name: str, num_classes: int, pretrained: bool = False) -> ResNet:
    """
    Creates a ResNet model with or without pretrained weights.

    :param model_name: Name of the ResNet variant (e.g., 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').
    :param num_classes: Number of output classes for classification.
    :param pretrained: Whether to load pretrained weights (default: False).
    :return: A ResNet model instance.
    """

    resnet_configs = {
        'resnet18': [BasicBlock, [2, 2, 2, 2]],  # block type, layers per block
        'resnet34': [BasicBlock, [3, 4, 6, 3]],
        'resnet50': [Bottleneck, [3, 4, 6, 3]],
        'resnet101': [Bottleneck, [3, 4, 23, 3]],
        'resnet152': [Bottleneck, [3, 8, 36, 3]],
    }

    if model_name not in resnet_configs:
        raise ValueError(f"Unknown ResNet variant: '{model_name}'. "
                         f"Available options: {list(resnet_configs.keys())}")

    block, layers = resnet_configs[model_name]
    model = ResNet(block, layers, num_classes)

    if pretrained:  # TODO würde überflüssig werden?
        if model_name == 'resnet18':
            pretrained_model = models.resnet18(weights="IMAGENET1K_V1")
        elif model_name == 'resnet34':
            pretrained_model = models.resnet34(weights="IMAGENET1K_V1")
        elif model_name == 'resnet50':
            pretrained_model = models.resnet50(weights="IMAGENET1K_V1")
        elif model_name == 'resnet101':
            pretrained_model = models.resnet101(weights="IMAGENET1K_V1")
        elif model_name == 'resnet152':
            pretrained_model = models.resnet152(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        pretrained_state = deepcopy(pretrained_model.state_dict())
        model.load_state_dict(pretrained_state, strict=False)

    return model
