import torch
import torch.nn as nn
from torch import Tensor
from torch.utils import model_zoo

from domgen.augment._mixstyle import MixStyle

model_urls = {
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169": "https://download.pytorch.org/models/densenet161-8d451a50.pth",
    "densenet201": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
}

class DenseNet(nn.Module):
    """DenseNet model with optional MixStyle insertion."""
    def __init__(
            self,
            growth_rate: int,
            block_layers: list,
            num_classes: int,
            mixstyle_layers: list = [],
            mixstyle_p: float = 0.5,
            mixstyle_alpha: float = 0.3,
            mix_type: str = "random",
            **kwargs
    ):
        super(DenseNet, self).__init__()
        self.mixstyle = None
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix=mix_type)
            print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

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
            self.features.add_module(
                f"DenseBlock_{i + 1}",
                DenseBlock(
                    in_channels,
                    growth_rate,
                    num_layers,
                    use_mixstyle=(f"DenseBlock_{i+1}" in mixstyle_layers),
                    mixstyle=self.mixstyle,
                )
            )
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
            num_layers,
            use_mixstyle = False,
            mixstyle = None
    ):
        super(DenseBlock, self).__init__()
        self.use_mixstyle = use_mixstyle
        self.mixstyle = mixstyle
        layers = []

        for i in range(num_layers):
            layers.append(self._make_layer(
                (in_channels + i * growth_rate),
                growth_rate,
                use_mixstyle=self.use_mixstyle,
                mixstyle=self.mixstyle)
            )
        self.block = nn.Sequential(*layers)

    @staticmethod
    def _make_layer(in_channels, growth_rate, use_mixstyle, mixstyle):
        layers = [
            nn.BatchNorm2d(in_channels)
        ]
        if use_mixstyle and mixstyle is not None:
            layers.append(mixstyle)  # mixstyle after batchnorm
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        ])
        return nn.Sequential(*layers)


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

def densenet121(growth_rate=32, num_classes=1000, pretrained=False, **kwargs):
    model = DenseNet(
        growth_rate=growth_rate, block_layers=[6, 12, 24, 16], pretrained=pretrained, num_classes=num_classes, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet169(growth_rate=32, pretrained=False , num_classes=1000, **kwargs):
    model = DenseNet(
        growth_rate=growth_rate, block_layers=[6, 12, 32, 32], pretrained=pretrained, num_classes=num_classes, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet169'])
    return model


def densenet201(growth_rate=32, pretrained=False , num_classes=1000, **kwargs):
    model = DenseNet(
        growth_rate=growth_rate, block_layers=[6, 12, 48, 32], pretrained=pretrained, num_classes=num_classes, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet264(growth_rate=32, pretrained=False , num_classes=1000, **kwargs):
    model = DenseNet(
        growth_rate=growth_rate, block_layers=[6, 12, 64, 48], pretrained=pretrained, num_classes=num_classes, **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model