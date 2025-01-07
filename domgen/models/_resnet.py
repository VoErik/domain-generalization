import torch.nn as nn
from torch import Tensor
from typing import Type

from torch.utils import model_zoo

from ._mixstyle import MixStyle


model_urls = {
    'resnet18':
        'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
        'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
        'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
        'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
        'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module):
    """
    ResNet model with integrated MixStyle as proposed in Zhou et al. (2021): https://arxiv.org/abs/2104.02008
    """
    def __init__(
            self,
            block: Type[nn.Module],
            layers: list,
            num_classes: int,
            mixstyle_layers: list = [],
            mixstyle_p: float = 0.5,
            mixstyle_alpha: float = 0.3,
            **kwargs
    ):
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

        self.mixstyle = None
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='random')
            print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

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

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer2(x)
        if 'layer2' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer3(x)
        if 'layer3' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer4(x)
        if 'layer4' in self.mixstyle_layers:
            x = self.mixstyle(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        out = self.feature_extractor(x)

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
        self.downsample = downsample
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


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def resnet18(num_classes: int, pretrained: bool = True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, mixstyle_layers=['layer1', 'layer2', 'layer3'], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet34(num_classes: int, pretrained: bool = True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model


def resnet50(num_classes: int, pretrained: bool = True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet101(num_classes: int, pretrained: bool = True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model


def resnet152(num_classes: int, pretrained: bool = True, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model
