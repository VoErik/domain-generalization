import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from copy import deepcopy


class DenseNet(nn.Module):
    def __init__(self, growth_rate: int, block_layers: list, num_classes: int):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        in_channels = 2 * growth_rate
        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_layers):
            self.features.add_module(f"DenseBlock_{i + 1}", DenseBlock(in_channels, growth_rate, num_layers))
            in_channels = in_channels + num_layers * growth_rate
            if i != len(block_layers) - 1:  # no transition after last block
                out_channels = in_channels // 2
                self.features.add_module(f"Transition_{i + 1}", TransitionLayer(in_channels, out_channels))
                in_channels = out_channels

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        out = self.init_conv(x)
        out = self.features(out)
        out = self.classifier(out)
        return out


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


class TransitionLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels
    ):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


def load_pretrained_weights(model: nn.Module, model_name: str):
    if model_name == 'densenet121':
        pretrained_model = models.densenet121(pretrained=True)
    elif model_name == 'densenet169':
        pretrained_model = models.densenet169(pretrained=True)
    elif model_name == 'densenet201':
        pretrained_model = models.densenet201(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} is not supported for loading pretrained weights.")

    pretrained_state = deepcopy(pretrained_model.state_dict())
    model.load_state_dict(pretrained_state, strict=False)
    return model


def densenet121_scratch():
    return DenseNet(growth_rate=32, block_layers=[6, 12, 24, 16], num_classes=1000)


def densenet169_scratch():
    return DenseNet(growth_rate=32, block_layers=[6, 12, 32, 32], num_classes=1000)


def densenet201_scratch():
    return DenseNet(growth_rate=32, block_layers=[6, 12, 48, 32], num_classes=1000)


# DenseNet264 isn't supported in Torchvision for ImageNet weights!
# TODO: trotzdem drinlassen?
def densenet264_scratch():
    return DenseNet(growth_rate=32, block_layers=[6, 12, 64, 48], num_classes=1000)


def densenet121():
    model = densenet121_scratch()
    model = load_pretrained_weights(model, "densenet121")
    return model


def densenet169():
    model = densenet169_scratch()
    model = load_pretrained_weights(model, "densenet169")
    return model


def densenet201():
    model = densenet201_scratch()
    model = load_pretrained_weights(model, "densenet201")
    return model


# DenseNet264 isn't supported in Torchvision for ImageNet weights!
# TODO: trotzdem drinlassen?
def densenet264():
    model = densenet264_scratch()
    model = load_pretrained_weights(model, "densenet264")
    return model
