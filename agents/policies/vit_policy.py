import numpy as np
import torch
import SemesterProject2.scripts.configs_network as configs_network
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.helpers.pytorch_utils as ptu

from monai.transforms import Resize
from torch.distributions import Normal
from SemesterProject2.agents.policies.base_policy import BasePolicy
from SemesterProject2.agents.vit_agent import ViTAgent


class ViTPolicy(BasePolicy):
    def __init__(self, agent: ViTAgent):
        super(ViTPolicy, self).__init__()
        self.agent = agent
        self.terminal = False
        self.obs_buffer = [None, None, None, None]  # list of ndarray
        self.resizer_small = Resize([configs_network.encoder_params["patch_size"]] * 3)
        self.resizer_large = Resize([configs_network.encoder_params["patch_size"] * 2] * 3)

    @torch.no_grad()
    def get_action(self, obs) -> np.ndarray:
        # obs: ((P_x, P_y, P_z), (2P_x, 2P_y, 2P_y), (3,), (3,))
        if_encoder_train = self.agent.encoder.training
        if_actor_train = self.agent.actor_head.training
        self.agent.encoder.eval()
        self.agent.actor_head.eval()

        patch_small, patch_large, center, size = obs
        patch_small = self.resizer_small(patch_small[None, ...])  # (1, P, P, P)
        patch_large = self.resizer_large(patch_large[None, ...])  # (1, 2P, 2P, 2P)
        # ((t, 1, P, P, P), (t, 1, 2P, 2P, 2P), (t, 3), (t, 3))
        self.add_to_buffer_(patch_small, patch_large, center, size)
        # ((t, 1, 1, P, P, P), (t, 1, 1, 2P, 2P, 2P), (t, 1, 3), (t, 1, 3))
        obs_buffer = [ptu.from_numpy(item).float().unsqueeze(1) for item in self.obs_buffer]
        obs_buffer[0] = 2 * obs_buffer[0] - 1
        obs_buffer[1] = 2 * obs_buffer[1] - 1

        embs = self.agent.encode_seq_(obs_buffer)  # (t, 1, N_emb)
        act = self.agent.actor_head(embs[-1:, ...]).squeeze()  # (1, 1, 8) -> (8,)
        mu, log_sigma = act[:3], act[3:6]  # both (3,)

        ### debugging only ###
        # mu, log_sigma = obs_buffer[2][-1, 0, :], ptu.from_numpy(np.array([1, 1, 1]))
        # mu, log_sigma = obs_buffer[2][-1, 0, :], log_sigma
        ### end of debugging block ###

        sigma = log_sigma.exp()
        distr = Normal(mu, sigma)
        next_center = distr.sample()  # (3,)
        next_size = sigma

        if if_encoder_train:
            self.agent.encoder.train()
        if if_actor_train:
            self.agent.actor_head.train()

        if self.terminal:
            self.obs_buffer = [None, None, None, None]

        next_center, next_size = ptu.to_numpy(next_center).astype(int), ptu.to_numpy(next_size).astype(int)
        next_size = np.maximum(next_size, 1)  ### heuristic
        # print(f"{next_center}, {next_size}")

        return next_center, next_size

    def add_to_buffer_(self, patch_small, patch_large, center, size):
        obs = (patch_small, patch_large, center, size)
        if self.obs_buffer[0] is None:
            self.obs_buffer[0] = patch_small[None, ...]  # (1, 1, P, P, P)
            self.obs_buffer[1] = patch_large[None, ...]
            self.obs_buffer[2] = center[None, ...]  # (1, 3)
            self.obs_buffer[3] = size[None, ...]

        elif self.obs_buffer[0].shape[0] < configs_ac.volumetric_env_params["num_steps_to_memorize"]:
            for i in range(len(self.obs_buffer)):
                self.obs_buffer[i] = np.concatenate([self.obs_buffer[i], obs[i][None, ...]], axis=0)

        else:
            for i in range(len(self.obs_buffer)):
               self.obs_buffer[i] = np.concatenate([self.obs_buffer[i][1:, ...], obs[i][None, ...]], axis=0)
