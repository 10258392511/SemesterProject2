import torch.nn as nn
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.helpers.pytorch_utils as ptu

from SemesterProject2.helpers.networks import network_initializer_ac
from SemesterProject2.agents.critics.base_critic import BaseCritic


class ACCritic(nn.Module, BaseCritic):
    def __init__(self, params):
        """
        params: **env_params
            bash_params: model_name
        """
        super(ACCritic, self).__init__()
        self.params = params
        _, net_params, _, opt_params = configs_ac.get_network_config(self.params["model_name"])
        self.critic_net = network_initializer_ac(self.params["model_name"], "critic")
        self.loss = opt_params["loss"]()
        self.opt = opt_params["constructor"](self.critic_net.parameters(), **opt_params["optimizer_config"])

    def forward(self, obs):
        # obs: (B, N_obs)
        v_vals = self.critic_net(obs)  # (B, 1)
        v_vals = v_vals.squeeze(-1)  # (B,)

        return v_vals

    def update(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        obs = ptu.from_numpy(ob_no)  # (B, N_obs)
        acts = ptu.from_numpy(ac_na)  # (B, n_act)
        next_obs = ptu.from_numpy(next_ob_no)  # (B, N_obs)
        rewards = ptu.from_numpy(re_n)  # (B,)
        terminals = ptu.from_numpy(terminal_n)  # (B,)

        for _ in range(self.params["num_target_updates"]):
            v_vals_next = self.forward(next_obs)  # (B,)
            target = rewards + self.params["gamma"] * v_vals_next * (1 - terminals)  # (B,)
            target = target.detach()

            for _ in range(self.params["num_grad_steps_per_target_update"]):
                v_vals = self.forward(obs)  # (B,)
                loss = self.loss(v_vals, target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        return {"critic_loss": loss.item()}
