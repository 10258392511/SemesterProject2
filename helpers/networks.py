import torch
import torch.nn as nn
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs as configs


def network_initializer(model_name):
    network_config, opt_config = configs.get_network_config(model_name)

    network = None
    if model_name == "LunarLander":
        network = MLP(**network_config).to(ptu.device)

    elif model_name == "LunarLanderImg":
        network = LunarLanderConv(**network_config).to(ptu.device)

    opt = opt_config["constructor"](network.parameters(), **opt_config["optimizer_config"])
    loss = opt_config["loss"]()

    return network, opt, loss


class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers=[64, 64, 64]):
        super(MLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        layers = []
        next_dim = self.obs_dim
        for i in range(len(hidden_layers)):
            cur_dim = next_dim
            next_dim = hidden_layers[i]
            layers.append(nn.Linear(cur_dim, next_dim))
            layers.append(nn.ReLU())
        cur_dim = next_dim
        layers.append(nn.Linear(cur_dim, self.act_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        # (B, N_obs) -> (B, N_act)
        # X[:, 0] *= 300
        # X[:, 1] *= 400
        return self.layers(X)


class LunarLanderConv(nn.Module):
    def __init__(self, act_dim, input_size=(3, 256, 256)):
        super(LunarLanderConv, self).__init__()
        self.input_size = input_size
        self.act_dim = act_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_size[0], 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 128, 128)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(64, self.act_dim, kernel_size=1)
        )

    def forward(self, X):
        # (B, 3, 256, 256) -> (B, N_act)
        X = self.conv_layers(X)  # (B, 64, 16, 16)
        X = X.mean(dim=[2, 3], keepdims=True)  # (B, 64, 1, 1)
        X = self.fc(X)  # (B, N_act, 1, 1)
        X = X.squeeze(dim=-1)
        X = X.squeeze(dim=-1)

        return X
