import torch
import torch.nn as nn

from SemesterProject2.helpers.modules.resnet_modules import AbstractBlock
from SemesterProject2.helpers.modules.vit_modules import PatchEmbed, Block


class ResNet(nn.Module):
    def __init__(self, in_channels, start_num_channels, num_classes, block_type: AbstractBlock):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.start_num_channels = start_num_channels
        self.num_classes = num_classes
        self.block_type = block_type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, start_num_channels, kernel_size=3, stride=2, padding=1),
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
        # X: (B, C_in, H, W)
        X = self.first_conv(X)  # down 2, C_start
        X = self.first_pool(X)  # down 4, C_start
        X = self.blocks(X)  # down 4 * 2 ^ 3 = 32, C_start * 8
        X = X.mean(dim=[2, 3], keepdims=True)  # (B, C_start * 8, 1, 1)
        X = self.head(X)  # (B, num_classes, 1, 1)

        return X.squeeze(-1).squeeze(-1)

    def init_modules_(self):
        pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ViT(nn.Module):
    def __init__(self, img_size=(1, 256, 256), patch_size=16, num_classes=10, emb_dim=512, depth=12, num_heads=8,
                 mlp_ratio=4, attn_p=0.1, proj_p=0.1):
        assert img_size[1] == img_size[2] and img_size[1] % patch_size == 0
        super(ViT, self).__init__()
        self.patch_emb = PatchEmbed(img_size, patch_size, emb_dim)
        num_patches = img_size[1] // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_token = nn.Parameter(torch.zeros(1, num_patches ** 2 + 1, emb_dim))
        self.drop = nn.Dropout(proj_p)
        blocks = [Block(emb_dim, num_heads, mlp_ratio, attn_p, proj_p) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, X):
        # X: (B, C, H, W)
        X = self.patch_emb(X)   # (B, N, N_emb)
        X = torch.cat([self.cls_token.expand(X.shape[0], -1, -1), X], dim=1)  # (1, 1, N_emb) -> (B, 1, N_emb) --> (B, N + 1, N_emb)
        X = X + self.pos_token  # (B, N + 1, N_emb) + (1, N + 1, N_emb) -> (B, N + 1, N_emb)
        X = self.drop(X)
        X = self.blocks(X)
        X = self.norm(X)  # (B, N + 1, N_emb)
        X = X[:, 0, :]  # (B, N_emb)
        X = self.head(X)  # (B, N_classes)

        return X
