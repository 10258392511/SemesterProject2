import torch
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_network as configs_network

from torch.distributions import Normal
from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.modules.vit_agent_modules import Encoder, MLPHead
from SemesterProject2.helpers.replay_buffer_vit import ReplayBuffer


class ViTAgent(BaseAgent):
    def __init__(self, params):
        """
        params:
        configs_ac.volumetric_env_params:
            num_target_steps, num_grad_steps_per_target_update, lam_cls, replay_buffer_size, dice_score_small_th, l2_tao
        bash:
            num_pre_train_updates, pre_train_batch_size, eval_interval
        As keys:
            encoder_params, *_head_params
        """
        super(ViTAgent, self).__init__()
        self.params = params
        self.encoder = Encoder(self.params["encoder_params"]).to(ptu.device)
        self.patch_pred_head = MLPHead(self.params["patch_pred_head_params"]).to(ptu.device)
        self.critic_head = MLPHead(self.params["critic_head_params"]).to(ptu.device)
        self.actor_head = MLPHead(self.params["actor_head_params"]).to(ptu.device)
        self.replay_buffer = ReplayBuffer(self.params)

        # TODO: 4 optimizers

    def train(self, paths) -> dict:
        """
        TODO: encoding -> compute total rewards -> (update critic -> actor -> patch_pred) -> merge info_dicts
        All heads are updated only after a complete rollout, so total rewards can (should) be (pre-)computed only once.
        """
        pass

    def pre_train(self) -> dict:
        pass

    def pre_train_(self):
        pass

    @torch.no_grad()
    def pre_eval_(self):
        pass

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        """
        Sample rollouts: list[Path]; delegated to .replay_buffer
        """
        paths = self.replay_buffer.sample_recent_rollouts(batch_size)

        return paths

    def update_critic(self, embs_encoded, next_embs_encoded, next_embs, has_seen_lesion,
                      total_rewards, terminals) -> dict:
        pass

    def update_actor(self, embs_encoded, next_embs_encoded, actions, has_seen_lesion,
                     total_rewards, terminals) -> dict:
        pass

    def update_patch_pred(self, novelty_loss):
        pass

    def encode_seq_(self, obs):
        """
        obs: ((T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3), (T, B, 3))
        """
        X_small, X_large, X_pos, _ = obs
        T, B = X_small.shape[:2]
        N_emb = self.params["encoder_params"]["d_model"]
        embs = torch.empty((T, B, N_emb), dtype=X_small.dtype, device=X_small.device)
        for t in range(T):
            start = max(t + 1 - self.params["num_steps_to_memorize"], 0)
            X_emb = self.encoder(X_small[start:t + 1, ...], X_large[start:t + 1, ...], X_pos[start:t + 1, ...])  # (t, B, N_emb)
            embs[t, ...] = X_emb[-1, ...]

        # (T, B, N_emb)
        return embs

    def encoding_(self, obs, next_obs):
        """
        obs / next_obs: ((T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3), (T, B, 3))
        """
        embs_encoded = self.encode_seq_(obs)  # (T, B, N_emb)
        if_training_encoder = self.encoder.training
        # print(if_training_encoder)
        with torch.no_grad():
            self.encoder.eval()
            next_embs = self.encoder.embed(*next_obs[:3])
            next_embs_encoded = self.encoder.transformer_encoder(next_embs)  # (T, B, N_emb)

        if if_training_encoder:
            self.encoder.train()

        # all (T, B, N_emb)
        return embs_encoded, next_embs_encoded, next_embs

    def compute_v_vals_(self, embs_encoded, next_emb_encoded):
        """
        embs_encoded, next_emb_encoded: both (T, B, N_emb)
        """
        v_vals_current = self.critic_head(embs_encoded)  # (T, B)
        if_training_critic_head = self.critic_head.training
        with torch.no_grad():
            self.critic_head.eval()
            v_vals_next = self.critic_head(next_emb_encoded)  # (T, B)

        if if_training_critic_head:
            self.critic_head.train()

        # both (T, B)
        return v_vals_current, v_vals_next

    def compute_likelihood_and_penalty_(self, embs_encoded, acts_in, has_seen_lesion, mode="all"):
        """
        Parameters
        ----------
        All: Tensor, already sent to device
        embs_encoded: (T, B, N_emb)
        acts: ((T, B, 3), (T, B, 3))
        has_seen_lesion: (T, B)

        Returns
        -------
        mode == "all":
            bbox_lh: (T, B)
            clf_reward: (T, B), float32
            clf_penalty: (1,)
        mode == "clf_only":
            clf_penalty: (1,)
        """
        assert mode in ("all", "clf_only"), "invalid mode"

        acts = self.actor_head(embs_encoded)  # (T, B, 8)
        mu, log_sigma, cls = acts[..., :3], acts[..., 3:6], acts[..., 6:]  # (T, B, 3), (T, B, 3), (T, B, 2)
        cls = torch.softmax(cls, dim=-1)  # (T, B, 2)
        # L2 distance for unbanlanced data; each: [has_seen_lesion, has_not_seen_legion]
        # (T, B) + (T, B) -.mean()-> (1,)
        clf_penalty = ((cls[..., 0] - has_seen_lesion) ** 2 + ((cls[..., 1] - (1 - has_seen_lesion))) ** 2).mean()

        clf_pred = cls[..., 0] > 0.5  # (T, B)
        with torch.no_grad():
            clf_reward = (clf_pred == has_seen_lesion).to(embs_encoded.dtype)  # (T, B)

        if mode == "clf_only":
            return clf_reward, clf_penalty

        normal_distr = Normal(mu, log_sigma.exp())  # (T, B, 3)
        bbox_lh = normal_distr.log_prob(acts_in[0]).sum(dim=-1)  # (T, B, 3) -> (T, B)

        return bbox_lh, clf_reward, clf_penalty

    def compute_novelty_seeking_reward_(self, embs_encoded, next_embs):
        """
        embs_encoded, next_embs: both (T, B, N_emb)
        """
        next_embs_pred = self.patch_pred_head(embs_encoded)  # (T, B, N_emb)
        next_embs = next_embs.detach()  # gurantee no gradient flow here
        l2_dist_squared = (next_embs_pred - next_embs) ** 2  # (T, B, N_emb)
        loss = l2_dist_squared.mean()
        # print(next_embs_pred)
        # print(next_embs)

        with torch.no_grad():
            # essentially Gaussian prior
            # (T, B, N_emb) -> (T, B)
            reward = 1 - torch.exp(-(l2_dist_squared.mean(dim=-1)) / self.params["l2_tao"])

        # (1,), (T, B)
        return loss, reward
