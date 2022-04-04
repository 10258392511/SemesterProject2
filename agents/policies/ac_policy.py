import numpy as np
import torch
import torch.nn as nn
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.helpers.pytorch_utils as ptu

# from itertools import chain
from torch.distributions import Normal
from SemesterProject2.agents.policies.base_policy import BasePolicy
from SemesterProject2.helpers.networks import network_initializer_ac


class ACPolicy(BasePolicy, nn.Module):
    def __init__(self, params):
        """
        params: **env_params,
            bash_args: model_name
        """
        super(ACPolicy, self).__init__()
        self.params = params
        actor_network_params, _, actor_opt_params, _ = configs_ac.get_network_config(self.params["model_name"])
        self.mean_net = network_initializer_ac(self.params["model_name"], "actor")
        self.log_std = nn.Parameter(torch.zeros(self.params["act_dim"], dtype=torch.float32).to(ptu.device))
        self.opt = actor_opt_params["constructor"](self.parameters(), **actor_opt_params["optimizer_config"])

    def save(self, filepath) -> None:
        torch.save(self.state_dict(), filepath)

    def forward(self, obs: torch.Tensor):
        # obs: (B, N_obs)
        act_mean = self.mean_net(obs)  # (B, N_act)

        # "(B, N_act)"
        return Normal(act_mean, self.log_std.exp())

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        act_distr = self.forward(obs)
        act = act_distr.sample()  # (B, N_act)

        return ptu.to_numpy(act)

    def update(self, obs: np.ndarray, acts: np.ndarray, adv: np.ndarray = None) -> dict:
        assert adv is not None
        obs = ptu.from_numpy(obs)  # (B, N_obs)
        acts = ptu.from_numpy(acts)  # (B, N_act)
        adv = ptu.from_numpy(adv)  # (B,)
        acts_distr = self.forward(obs)  # "(B, N_act)"
        log_prob = acts_distr.log_prob(acts)  # (B, N_act)
        log_prob = log_prob.sum(dim=1)  # (B,)
        loss = -(log_prob * adv).sum()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"actor_loss": loss.item()}
