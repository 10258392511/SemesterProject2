import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(1, 256, 256), patch_size=16, emb_dim=512):
        assert img_size[1] == img_size[2] and img_size[1] % patch_size == 0
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_one_dim = self.img_size[1] // self.patch_size
        self.embed_dim = emb_dim
        self.proj = nn.Conv2d(self.img_size[0], self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, X):
        # X: (B, C, H, W)
        X = self.proj(X)  # (B, N_emb, N_patches, N_patches)
        X = X.reshape((*X.shape[:2], -1))  # (B, N_emb, N_patches ** 2)
        X = X.permute(0, 2, 1)  # (B, N_patches ** 2, N_emb)

        return X


class Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, attn_drop_p=0.1, proj_drop_p=0.1):
        assert emb_dim % num_heads == 0
        super(Attention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = self.emb_dim // self.num_heads
        self.qkv = nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.scale = 1 / np.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_p)
        self.proj_drop = nn.Dropout(proj_drop_p)
        self.proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, X):
        # X: (B, N, N_emb)
        X = self.qkv(X)  # (B, N, 3 * N_emb)
        X = X.reshape(X.shape[0], X.shape[1], 3, self.num_heads, self.head_dim)  # (B, N, 3, N_heads, head_dim)
        X = X.permute(0, 2, 3, 1, 4)  # (B, 3, N_heads, N, head_dim)
        q, k, v = X[:, 0, ...], X[:, 1, ...], X[:, 2, ...]  # (B, N_heads, N, head_dim) each
        k_t = k.permute(0, 1, 3, 2)  # (B, N_heads, head_dim, N)
        dot_p = q @ k_t  # (B, N_heads, N, N)
        dot_p = dot_p * self.scale
        attn = F.softmax(dot_p, dim=-1)  # (B, N_heads, N, N)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v  # (B, N_heads, N, N) @ (B, N_heads, N, head_dim) -> (B, N_heads, N, head_dim)
        weighted_avg = weighted_avg.permute(0, 2, 1, 3)  # (B, N, N_heads, head_dim)
        weighted_avg = weighted_avg.reshape(weighted_avg.shape[0], weighted_avg.shape[1], -1)  # (B, N, N_emb)
        X = self.proj(weighted_avg)  # (B, N, N_emb)
        X = self.proj_drop(X)

        # X: (B, N, N_emb)
        return X


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, p=0.1):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Dropout(p)
        )

    def forward(self, X):
        # X: (B, N, N_emb) -> (B, N, N_emb)

        return self.layers(X)


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio, attn_p=0.1, proj_p=0.1):
        super(Block, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.attn = Attention(self.emb_dim, self.num_heads, attn_p, proj_p)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.mlp = MLP(self.emb_dim, self.emb_dim, self.mlp_ratio * self.emb_dim, proj_p)

    def forward(self, X):
        # X: (B, N, N_emb) -> (B, N, N_emb)
        X_out = self.norm1(X)
        X_out = self.attn(X_out)
        X = X + X_out
        X_out = self.norm2(X)
        X_out = self.mlp(X_out)
        X = X + X_out

        return X
