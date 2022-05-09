import torch
import torch.nn as nn
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
            num_target_steps, num_grad_steps_per_target_update, lam_cls, replay_buffer_size,
            dice_score_small_th, l2_tao, gamma, if_clip_grad
        As keys: configs_networks:
            encoder_params, *_head_params,
            encoder_opt_args, *_head_opt_args
        """
        super(ViTAgent, self).__init__()
        self.params = params
        self.encoder = Encoder(self.params["encoder_params"]).to(ptu.device)
        # TODO: load pre-trained Encoder
        self.patch_pred_head = MLPHead(self.params["patch_pred_head_params"]).to(ptu.device)
        self.critic_head = MLPHead(self.params["critic_head_params"]).to(ptu.device)
        self.actor_head = MLPHead(self.params["actor_head_params"]).to(ptu.device)
        self.replay_buffer = ReplayBuffer(self.params)
        self.encoder_opt = self.params["encoder_opt_args"]["class"](self.encoder.parameters(),
                                                                    **self.params["encoder_opt_args"]["args"])
        self.patch_pred_head_opt = self.params["patch_pred_head_opt_args"]["class"](self.patch_pred_head.parameters(),
                                                                    **self.params["patch_pred_head_opt_args"]["args"])
        self.critic_head_opt = self.params["critic_head_opt_args"]["class"](self.critic_head.parameters(),
                                                                    **self.params["critic_head_opt_args"]["args"])
        self.actor_head_opt = self.params["actor_head_opt_args"]["class"](self.actor_head.parameters(),
                                                                    **self.params["actor_head_opt_args"]["args"])

    def train(self, paths) -> dict:
        """
        encoding -> compute total rewards -> (update critic -> actor -> patch_pred) -> merge info_dicts
        All heads are updated only after a complete rollout, so total rewards can (should) be (pre-)computed only once.

        after .unsqueeze(1), sending to device, (2 * X - 1):
        obs, next_obs: ((T, 1, 1, P, P, P), (T, 1, 1, 2P, 2P, 2P), (T, 1, 3), (T, 1, 3))
        acts: ((T, 1, 3), (T, 1, 3))
        rewards: (T, 1)
        terminals: (T, 1)
        has_seen_lesion: (T, 1)
        """
        # .unsqueeze(1) and send to device (and 2 * X - 1) for all
        info_dict = {}  # returns info for the last path
        for path in paths:
            self.encoder.train()
            self.critic_head.train()
            self.actor_head.train()
            self.patch_pred_head.train()
            obs, acts, rewards, next_obs, terminals, has_seen_lesion = self.unsqueeze_and_send_to_device_(path)
            # all (T, B, N_emb)
            embs_encoded, next_embs_encoded, next_embs = self.encoding_(obs, next_obs)
            # compute total_reward: (T, 1)
            with torch.no_grad():
                self.actor_head.eval()
                self.patch_pred_head.eval()
                # (T, 1)
                clf_reward, _ = self.compute_likelihood_and_penalty_(embs_encoded, acts, has_seen_lesion,
                                                                     mode="clf_only")
                # (T, 1)
                _, novelty_reward = self.compute_novelty_seeking_reward_(embs_encoded, next_embs)
                total_reward = (clf_reward + novelty_reward).detach()

            self.actor_head.train()
            self.patch_pred_head.train()

            self.encoder_opt.zero_grad()
            # print("updating critic...")
            info_critic = self.update_critic(embs_encoded, next_embs_encoded, next_embs, has_seen_lesion,
                                             total_reward, terminals)
            # print("updating actor...")
            info_actor = self.update_actor(embs_encoded, next_embs_encoded, acts, has_seen_lesion,
                                           total_reward, terminals)
            # print("updating patch_pred...")
            info_patch_pred = self.update_patch_pred(embs_encoded, next_embs)
            if self.params["if_clip_grad"]:
                nn.utils.clip_grad_value_(self.encoder.parameters(),
                                          self.params["encoder_opt_args"]["clip_grad_val"])
            self.encoder_opt.step()

            # for dict_iter in (info_critic, info_actor, info_patch_pred):
            #     print(dict_iter)
            # print("-" * 100)

        for dict_iter in (info_critic, info_actor, info_patch_pred):
            info_dict.update(dict_iter)

        return info_dict

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
        # embs_encoded, next_embs_encoded, next_embs: (T, 1, N_emb), has_seen_lesion, total_rewards, terminals: (T, 1)
        for _ in range(self.params["num_target_updates"]):
            with torch.no_grad():
                self.critic_head.eval()
                # (T, 1)
                _, v_vals_next = self.compute_v_vals_(embs_encoded, next_embs_encoded)
                # (T,)
                targets = (total_rewards + self.params["gamma"] * v_vals_next * (1 - terminals)).squeeze().detach()

            self.critic_head.train()
            for _ in range(self.params["num_grad_steps_per_target_update"]):
                # (T, 1)
                v_vals_cur, _ = self.compute_v_vals_(embs_encoded, next_embs_encoded)
                loss = ((v_vals_cur.squeeze() - targets) ** 2).mean()
                # self.encoder_opt.zero_grad()
                self.critic_head_opt.zero_grad()
                loss.backward(retain_graph=True)  # retain computational graph from input to output of .encoder
                if self.params["if_clip_grad"]:
                    nn.utils.clip_grad_value_(self.critic_head.parameters(),
                                              self.params["critic_head_opt_args"]["clip_grad_val"])
                # self.encoder_opt.step()
                self.critic_head_opt.step()

            return {"critic_loss": loss.item()}

    def update_actor(self, embs_encoded, next_embs_encoded, actions, has_seen_lesion,
                     total_rewards, terminals) -> dict:
        # embs_encoded, next_emb_encoded: (T, 1, N_emb), actions: ((T, 1, 3), (T, 1, 3)),
        # has_seen_lesion, total_rewards, terminals: (T, 1)
        # (T, 1), (1,)
        bbox_lh, _, clf_penalty = self.compute_likelihood_and_penalty_(embs_encoded, actions, has_seen_lesion,
                                                                       mode="all")
        if_critic_train = self.critic_head.training
        with torch.no_grad():
            self.critic_head.eval()
            v_vals_cur, v_vals_next = self.compute_novelty_seeking_reward_(embs_encoded, next_embs_encoded)

        if if_critic_train:
            self.critic_head.train()

        # (T, 1)
        adv = (total_rewards + self.params["gamma"] * v_vals_next * (1 - terminals)).detach()
        nll = -(bbox_lh * adv).sum()
        loss = nll + clf_penalty * self.params["lam_cls"]

        # self.encoder_opt.zero_grad()
        self.actor_head_opt.zero_grad()
        loss.backward(retain_graph=True)  # retain computational graph from input to output of .encoder
        if self.params["if_clip_grad"]:
            nn.utils.clip_grad_value_(self.actor_head.parameters(),
                                      self.params["actor_head_opt_args"]["clip_grad_val"])
        # self.encoder_opt.step()
        self.actor_head_opt.step()

        return {
            "actor_nll": nll.item(),
            "actor_clf": clf_penalty.item(),
            "actor_loss": loss.item()
        }

    def update_patch_pred(self, embs_encoded, next_embs):
        # embs_encoded, next_embs: (T, 1, N_emb)
        # (1,)
        novelty_loss, _ = self.compute_novelty_seeking_reward_(embs_encoded, next_embs)
        # self.encoder_opt.zero_grad()
        self.patch_pred_head_opt.zero_grad()
        novelty_loss.backward(retain_graph=False)  # last update step
        if self.params["if_clip_grad"]:
            nn.utils.clip_grad_value_(self.patch_pred_head.parameters(),
                                      self.params["patch_pred_head_opt_args"]["clip_grad_val"])
        # self.encoder.step()
        self.patch_pred_head_opt.step()

        return {"novelty_loss": novelty_loss.item()}

    def encode_seq_(self, obs):
        """
        obs: ((T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3), (T, B, 3)), already sent to device
        """
        X_small, X_large, X_pos, _ = obs
        X_small = 2 * X_small - 1
        X_large = 2 * X_large - 1
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

    def unsqueeze_and_send_to_device_(self, path):
        """
        Also: 2 * X - 1 for patches

        Returns
        -------
        obs, next_obs: ((T, 1, 1, P, P, P), (T, 1, 1, 2P, 2P, 2P), (T, 1, 3), (T, 1, 3))
        acts: ((T, 1, 3), (T, 1, 3))
        rewards: (T, 1)
        terminals: (T, 1)
        has_seen_lesion: (T, 1)
        """
        obs = [ptu.from_numpy(item).unsqueeze(1) for item in path["observations"]]
        obs[0] = 2 * obs[0] - 1
        obs[1] = 2 * obs[1] - 1

        next_obs = [ptu.from_numpy(item).unsqueeze(1) for item in path["next_obs"]]
        next_obs[0] = 2 * next_obs[0] - 1
        next_obs[1] = 2 * next_obs[1] - 1

        acts = [ptu.from_numpy(item).unsqueeze(1) for item in path["actions"]]
        rewards = ptu.from_numpy(path["rewards"]).unsqueeze(1)
        terminals = ptu.from_numpy(path["terminals"]).unsqueeze(1)
        has_seen_lesion = ptu.from_numpy(path["infos"]["has_seen_lesion"]).unsqueeze(1)

        return obs, acts, rewards, next_obs, terminals, has_seen_lesion

    def load_encoder(self, filename):
        assert self.encoder is not None
        self.encoder.load_state_dict(torch.load(filename))
        self.encoder.eval()

    def load_heads(self, critic_filename, actor_filename, patch_pred_filename):
        assert self.critic_head is not None
        assert self.actor_head is not None
        assert self.patch_pred_head is not None

        for module_iter, filename_iter in zip((self.critic_head, self.actor_head, self.patch_pred_head),
                                           (critic_filename, actor_filename, patch_pred_filename)):
            module_iter.load_state_dict(torch.load(filename_iter))
            module_iter.eval()
