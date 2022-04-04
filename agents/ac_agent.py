import numpy as np
import torch
import SemesterProject2.helpers.pytorch_utils as ptu

from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.replay_buffer import ReplayBuffer
from SemesterProject2.agents.policies.ac_policy import ACPolicy
from SemesterProject2.agents.critics.ac_critic import ACCritic


class ACAgent(BaseAgent):
    def __init__(self, params):
        """
        params: **env_params, **bash_params
        """
        super(ACAgent, self).__init__()
        self.params = params
        self.actor = ACPolicy(self.params)
        self.critic = ACCritic(self.params)
        self.replay_buffer = ReplayBuffer(params["replay_buffer_size"])

    def train(self, obs, acts, next_obs, rewards, terminals) -> dict:
        loss_dict = {}
        for _ in range(self.params["num_critic_updates_per_agent_update"]):
            loss_critic_dict = self.critic.update(obs, acts, next_obs, rewards, terminals)

        loss_dict.update(loss_critic_dict)
        adv = self.estimate_adv_(obs, next_obs, rewards, terminals)
        for _ in range(self.params["num_actor_updates_per_agent_update"]):
            loss_actor_dict = self.actor.update(obs, acts, adv)

        loss_dict.update(loss_actor_dict)

        # {"critic_loss": float, "actor_loss": float}
        return loss_dict

    @torch.no_grad()
    def estimate_adv_(self, obs, next_obs, rewards, terminals, eps=1e-5) -> np.ndarray:
        obs = ptu.from_numpy(obs)  # (B, N_obs)
        next_obs = ptu.from_numpy(next_obs)  # (B, N_obs)
        rewards = ptu.from_numpy(rewards)  # (B,)
        terminals = ptu.from_numpy(terminals)  # (B,)

        v_vals = self.critic(obs)  # (B,)
        q_vals = rewards + self.params["gamma"] * self.critic(next_obs) * (1 - terminals)  # (B,)
        adv = ptu.to_numpy(q_vals - v_vals)  # (B,)
        adv = (adv - np.mean(adv)) / (np.std(adv) + eps)

        return adv

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
