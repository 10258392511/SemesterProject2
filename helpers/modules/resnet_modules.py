import torch
import torch.nn as nn
import torch.nn.functional as F
import abc


class AbstractBlock(nn.Module, abc.ABC):
    def __init__(self, in_channels, out_channels):
        super(AbstractBlock, self).__init__()
        self.in_channels = in_channels  # for debug
        self.out_channels = out_channels
        self.layers = None
        self.downsample = None

    def forward(self, X):
        identity = X.clone()
        X = self.layers(X)
        identity = self.downsample(identity)

        return F.relu(X + identity)


class BasicBlock(AbstractBlock):
    def __init__(self, in_channels, out_channels, if_downsample=False):
        super(BasicBlock, self).__init__(in_channels, out_channels)

        stride = 2 if if_downsample else 1
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )



class BottleneckBlock(AbstractBlock):
    def __init__(self, in_channels, out_channels, if_downsample=False):
        assert in_channels % 2 == 0 and out_channels % 2 == 0
        super(BottleneckBlock, self).__init__(in_channels, out_channels)
        stride = 2 if if_downsample else 1
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
