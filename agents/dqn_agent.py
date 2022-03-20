import numpy as np
import torch
import torch.nn as nn
import SemesterProject2.scripts.configs as configs
import SemesterProject2.helpers.pytorch_utils as ptu

from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.networks import network_initializer
from SemesterProject2.helpers.replay_buffer import ReplayBuffer
from SemesterProject2.helpers.utils import Path


class DQNAgent(BaseAgent):
    def __init__(self, hparams):
        """
        hparams: bash params: if_double_q, network_name
                 env params: grad_clamp_val
        """
        self.hparams = hparams
        self.q_net, self.optimizer, self.loss = network_initializer(self.hparams["model_name"])
        self.q_net_target, _, _ = network_initializer(self.hparams["model_name"])
        for param, param_target in zip(self.q_net.parameters(), self.q_net_target.parameters()):
            param_target.data.copy_(param.data)
        self.q_net_target.eval()

        self.env_params = configs.get_env_config(self.hparams["model_name"])
        self.replay_buffer = ReplayBuffer(self.env_params["replay_buffer_size"])
        self.t = 0
        # self.loss = nn.SmoothL1Loss()

    def train(self, obs, act, reward, next_obs, terminal):
        paths = [Path([obs], [], [act], [reward], [next_obs], [bool(terminal)])]
        self.replay_buffer.add_rollouts(paths)
        self.t += 1
        log = {}
        if self.t % self.env_params["update_freq"] == 0:
            if self.replay_buffer.can_sample(self.env_params["batch_size"]):
                obs_sampled, acts_sampled, rewards_sampled, next_obs_sampled, terminals_sampled = self.replay_buffer.sample_random_data(
                    self.env_params["batch_size"])
                log = self.update(obs_sampled, acts_sampled, rewards_sampled, next_obs_sampled, terminals_sampled)

        return log

    def update(self, obs, acts, rewards, next_obs, terminals):
        self.q_net.train()

        obs = ptu.from_numpy(obs)  # (B, N_obs) or (B, C, H, W)
        acts = ptu.from_numpy(acts)  # (B,), only works for discrete actions
        rewards = ptu.from_numpy(rewards)  # (B,)
        next_obs = ptu.from_numpy(next_obs)  # (B, N_obs) or (B, C, H, W)
        terminals = ptu.from_numpy(terminals)  # (B,)

        qa_vals = self.q_net(obs)  # (B, N_act)
        est_current = torch.gather(qa_vals, 1, acts.unsqueeze(1)).squeeze()  # (B,) -> (B, 1), (B, 1) -> (B,)
        qa_vals_next_target = self.q_net_target(next_obs)

        if self.hparams["if_double_q"]:
            self.q_net.eval()
            qa_vals_next = self.q_net(next_obs)  # (B, N_act)
            acts_next = qa_vals_next.argmax(dim=1, keepdim=True)  # (B, 1)
            est_next = torch.gather(qa_vals_next_target, 1, acts_next).squeeze()  # (B,)
        else:
            est_next, _ = qa_vals_next_target.max(dim=1)

        labels_current = rewards + self.env_params["gamma"] * (1 - terminals) * est_next
        labels_current = labels_current.detach()

        self.q_net.train()
        loss = self.loss(est_current, labels_current)
        self.optimizer.zero_grad()
        loss.backward()
        if "grad_clamp_val" in self.env_params:
            # print("clamping gradient")
            nn.utils.clip_grad_value_(self.q_net.parameters(), self.hparams["grad_clamp_val"])
        self.optimizer.step()

        tau = self.env_params["tau"]
        for param, param_target in zip(self.q_net.parameters(), self.q_net_target.parameters()):
            param_target.data.copy_(param.data * tau + param_target.data * (1 - tau))

        return {"training_loss": loss.item()}

    @torch.no_grad()
    def get_action(self, obs, eps):
        sample = np.random.rand()
        obs = obs[None, ...]  # (N_obs,) -> (1, N_obs)
        obs = ptu.from_numpy(obs)
        qa_vals = self.q_net(obs).squeeze()  # (1, N_act) -> (N_act,)
        qa_vals = ptu.to_numpy(qa_vals)
        if sample <= eps:
            return np.random.randint(0, qa_vals.shape[0])
        return qa_vals.argmax(axis=-1)

    def save(self, filename):
        torch.save(self.q_net.state_dict(), filename)

    def load(self, filename):
        self.q_net.load_state_dict(torch.load(filename))
        self.q_net.eval()
