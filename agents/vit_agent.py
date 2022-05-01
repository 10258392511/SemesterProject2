import torch
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_network as configs_network

from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.modules.vit_agent_modules import Encoder, MLPHead
from SemesterProject2.helpers.replay_buffer_vit import ReplayBuffer


class ViTAgent(BaseAgent):
    def __init__(self, params):
        """
        params:
        configs_ac.volumetric_env_params:
            num_target_steps, num_grad_steps_per_target_update, lam_cls, replay_buffer_size, dice_score_small_th, l2_tao
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

    def train(self, paths) -> dict:
        pass

    def pre_train(self, paths) -> dict:
        pass

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        """
        Sample rollouts: list[Path]; delegated to .replay_buffer
        """
        paths = self.replay_buffer.sample_recent_rollouts(batch_size)

        return paths

    def update_critic(self, paths) -> dict:
        pass

    def update_actor(self, paths) -> dict:
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

    def compute_likelihood_and_penalty_(self, paths):
        """
        Returns:
            bbox_log_lh: (T,), cls_log_lh: (T,), cls_penalty: (1,)
        """
        pass

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
