import torch.nn as nn
from DenseBlock import DenseBlock
from TransitionLayer import TransitionLayer


class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_layers, num_classes):
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
            nn.Softmax(dim=1)  # mit rein oder durch loss function einbezogen?
        )

    def forward(self, x):
        out = self.init_conv(x)
        out = self.features(out)
        out = self.classifier(out)
        return out

    @staticmethod
    def densenet121():
        return DenseNet(growth_rate=32, block_layers=[6, 12, 24, 16], num_classes=1000)

    @staticmethod
    def densenet169():
        return DenseNet(growth_rate=32, block_layers=[6, 12, 32, 32], num_classes=1000)

    @staticmethod
    def densenet201():
        return DenseNet(growth_rate=32, block_layers=[6, 12, 48, 32], num_classes=1000)

    @staticmethod
    def densenet264():
        return DenseNet(growth_rate=32, block_layers=[6, 12, 64, 48], num_classes=1000)
