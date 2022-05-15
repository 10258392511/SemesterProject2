import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, params):
        """
        params: configs.encoder_params
            in_channels, d_model, patch_size, nhead, dim_feedforward, dropout, num_layers, if_pos_enc
        hparams: if_pos_enc
        """
        super(Encoder, self).__init__()
        self.params = params
        self.patch_embed = nn.Conv3d(in_channels=self.params["in_channels"] * 9, out_channels=self.params["d_model"],
                                     kernel_size=self.params["patch_size"], stride=self.params["patch_size"])
        if self.params["if_pos_enc"]:
            self.pos_emb = nn.Linear(3, self.params["d_model"])
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.params["d_model"],
                                                                                    nhead=self.params["nhead"],
                                                                                    dim_feedforward=self.params["dim_feedforward"],
                                                                                    dropout=self.params["dropout"]),
                                                         num_layers=self.params["num_layers"],
                                                         norm=nn.LayerNorm(self.params["d_model"]))

    def forward(self, X_small: torch.Tensor, X_large: torch.Tensor, X_pos: torch.Tensor):
        X = self.embed(X_small, X_large, X_pos)
        X = self.transformer_encoder(X)  # (T, B, d_model)

        return X

    def embed(self,  X_small: torch.Tensor, X_large: torch.Tensor, X_pos: torch.Tensor):
        # X_small: (T, B, 1, P, P, P), X_large: (T, B, 1, 2P, 2P, 2P), X_pos: (T, B, 3)
        T, B, C_in, P = X_small.shape[:4]
        assert P == self.params["patch_size"] and C_in == self.params["in_channels"]
        X_pos = X_pos.to(X_small.dtype)
        if self.params["if_pos_enc"]:
            X_pos = self.pos_emb(X_pos)  # (T, B, d_model)
        X_large = X_large.reshape((T, B, C_in, 2, P, 2, P, 2, P))
        X_large = X_large.permute((0, 1, 2, 3, 5, 7, 4, 6, 8))  # (T, B, C_in, 2, 2, 2, P, P, P)
        X_large = X_large.reshape(T, B, -1, P, P, P)  # (T, B, C_in * 8, P, P, P)
        X = torch.cat([X_small, X_large], dim=2)  # (T, B, 9 * C_in, P, P, P)
        X = X.reshape(-1, 9 * C_in, P, P, P)  # (T * B, 9 * C_in, P, P, P)
        X = self.patch_embed(X).squeeze()  # (T * B, d_model, 1, 1, 1) -> (T * B, d_model)
        X = X.reshape((T, B, -1))  # (T, B, d_model)
        if self.params["if_pos_enc"]:
            X = X + X_pos  # (T, B, d_model)

        return X


class MLPHead(nn.Module):
    def __init__(self, params):
        """
        params: configs.*_head_params
            in_features, out_features, hidden_layers: tuple, last_act: act_func
        """
        super(MLPHead, self).__init__()
        self.params = params
        layer_dims = [self.params["in_features"]] + list(self.params["hidden_layers"]) + [self.params["out_features"]]
        layers = []
        for in_features_iter, out_features_iter in zip(layer_dims, layer_dims[1:]):
            layers.append(nn.Linear(in_features_iter, out_features_iter))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        # X: (B, d_model)
        X = self.layers(X)
        X = self.params["last_act"](X)

        if X.shape[-1] == 1:
            X = X.squeeze()  # (B, 1) -> (B,) (for critic_head)
        return X
