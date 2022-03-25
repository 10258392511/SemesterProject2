import torch.nn as nn

from SemesterProject2.helpers.modules.resnet_modules import AbstractBlock


class ResNet(nn.Module):
    def __init__(self, in_channels, start_num_channels, num_classes, block_type: AbstractBlock):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.start_num_channels = start_num_channels
        self.num_classes = num_classes
        self.block_type = block_type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, start_num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(start_num_channels),
            nn.ReLU()
        )
        self.first_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            block_type(start_num_channels, start_num_channels * 2),
            block_type(start_num_channels * 2, start_num_channels * 2, if_downsample=True),
            block_type(start_num_channels * 2, start_num_channels * 4),
            block_type(start_num_channels * 4, start_num_channels * 4, if_downsample=True),
            block_type(start_num_channels * 4, start_num_channels * 8),
            block_type(start_num_channels * 8, start_num_channels * 8, if_downsample=True)
        )
        self.head = nn.Conv2d(start_num_channels * 8, num_classes, kernel_size=1)
        self.init_modules_()

    def forward(self, X):
        # X: (B, Cin, H, W)
        X = self.first_conv(X)  # down 2, C_start
        X = self.first_pool(X)  # down 4, C_start
        X = self.blocks(X)  # down 4 * 2 ^ 3 = 32, C_start * 8
        X = X.mean(dim=[2, 3], keepdims=True)  # (B, C_start * 8, 1, 1)
        X = self.head(X)  # (B, num_classes, 1, 1)

        return X.squeeze(-1).squeeze(-1)

    def init_modules_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
