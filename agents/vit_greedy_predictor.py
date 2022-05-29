import numpy as np
import torch
import torch.nn as nn
import os

from SemesterProject2.agents.vit_greedy_agent import ViTGreedyAgent
from SemesterProject2.envs.volumetric import VolumetricForGreedy
from SemesterProject2.helpers.utils import center_size2start_end


class ViTGreedyPredictor(object):
    def __init__(self, params):
        """
        params: configs_ac.volumetric_env_params, configs_network.*_params
        bash:
            param_paths: {"encoder": "*.pt", "clf_head": ..., "patch_pred_head": ...}
            mode: "cartesion" or "explore"
            order: "zyx"...
            grid_size: (3, 3)
        """
        self.params = params
        self.env = VolumetricForGreedy(params)
        self.agent = ViTGreedyAgent(params)
        self.agent.load_encoder(self.params["param_paths"]["encoder"])
        self.agent.load_heads(self.params["param_paths"]["clf_head"],
                              self.params["param_paths"]["patch_pred_head"])
        self.trajectory = None  # [(X_pos, terminal)]
        self.obs_buffer = {
            "patches_small": None,
            "patches_large": None,
            "rel_pos": None
        }
        self.init_pos_grid = self.init_grid_()
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def predict(self):
        pass

    @torch.no_grad()
    def evaluate(self, bboxes):
        pass

    @torch.no_grad()
    def get_action(self, obs):
        X_small, X_large, X_pos, X_size = obs
        self.add_to_buffer_(X_small, X_large, X_pos)

        next_novelty = []
        for act in self.action_space:
            X_pos_next_candidate = (X_pos + act * X_size).astype(int)
            # print(X_pos_next_candidate)
            if (not self.env.is_in_vol_(X_pos_next_candidate, X_size * 2)) or \
                    self.recently_visited_(X_pos_next_candidate):
                novelty = 0
            else:
                novelty = self.compute_novelty_(X_pos_next_candidate)
            next_novelty.append(novelty)
        next_novelty = np.array(next_novelty)
        ind = np.argmax(next_novelty)

        next_center = X_pos + self.action_space[ind] * X_size
        next_size = X_size

        return next_center.astype(int), next_size

    def init_grid_(self):
        pass

    def add_to_buffer_(self, X_small, X_large, X_pos):
        """
        All: ndarray
        X_small: (P, P, P), X_large: (2P, 2P, 2P), X_pos: (3,)
        """
        X_pos = convert_to_rel_pos(X_pos, np.array(self.env.vol.shape[::-1]))
        X_small = X_small[None, None, None, ...]  # (1, 1, 1, P, P, P)
        X_large = X_large[None, None, None, ...]  # (1, 1, 1, 2P, 2P, 2P)
        X_pos = X_pos[None, None, ...]  # (1, 1, 3)
        if self.obs_buffer["rel_pos"] is None:
            self.obs_buffer["patches_small"] = X_small
            self.obs_buffer["patches_large"] = X_large
            self.obs_buffer["rel_pos"] = X_pos
        else:
            if self.obs_buffer["rel_pos"].shape[0] < self.params["num_steps_to_memorize"]:
                self.obs_buffer["patches_small"] = np.concatenate([self.obs_buffer["patches_small"], X_small], axis=0)
                self.obs_buffer["patches_large"] = np.concatenate([self.obs_buffer["patches_large"], X_large], axis=0)
                self.obs_buffer["rel_pos"] = np.concatenate([self.obs_buffer["rel_pos"], X_pos], axis=0)
            else:
                self.obs_buffer["patches_small"] = np.concatenate([self.obs_buffer["patches_small"][1:, ...], X_small],
                                                                  axis=0)
                self.obs_buffer["patches_large"] = np.concatenate([self.obs_buffer["patches_large"][1:, ...], X_large],
                                                                  axis=0)
                self.obs_buffer["rel_pos"] = np.concatenate([self.obs_buffer["rel_pos"][1:, ...], X_pos], axis=0)

    def clear_buffer_(self):
        for key in self.obs_buffer:
            self.obs_buffer[key] = None

    def compute_novelty_(self, center):
        center_original = center.copy()
        center = convert_to_rel_pos(center, np.array(self.env.vol.shape[::-1]))
        X_small, X_large, X_pos = self.obs_buffer["patches_small"], self.obs_buffer["patches_large"], \
                                  self.obs_buffer["rel_pos"]
        X_small = ptu.from_numpy(X_small) * 2 - 1
        X_large = ptu.from_numpy(X_large) * 2 - 1
        X_pos = ptu.from_numpy(X_pos)
        X_pos_next = ptu.from_numpy(center).reshape(1, 1, -1)  # (3,) -> (1, 1, 3)
        X_emb_enc = self.agent.encoder(X_small, X_large, X_pos, X_pos_next)  # (1, N_emb)
        X_next_pred = self.agent.patch_pred_head(X_emb_enc).squeeze()  # (1, N_emb) -> (N_emb,)
        # (P, P, P)
        X_next_candidate_small = self.env.get_patch_by_center_size(center_original,
                                                                   np.array(self.obs_buffer["patches_small"].shape[3:]))
        # (2P, 2P, 2P)
        X_next_candidate_large = self.env.get_patch_by_center_size(center_original,
                                                                   np.array(self.obs_buffer["patches_large"].shape[3:]))
        # (1, 1, 1, P, P, P), (1, 1, 1, 2P, 2P, 2P)
        X_next_candidate_small = ptu.from_numpy(X_next_candidate_small).reshape(1, 1, 1, *X_next_candidate_small.shape) * 2 - 1
        X_next_candidate_large = ptu.from_numpy(X_next_candidate_large).reshape(1, 1, 1, *X_next_candidate_large.shape) * 2 - 1
        X_next_patch_emb = self.encoder.emb_patch(X_next_candidate_small, X_next_candidate_large).squeeze()  # (1, 1, N_emb) -> (N_emb)
        novelty = self.mse(X_next_pred, X_next_patch_emb)

        return novelty.item()

    def recently_visited_(self, X_pos):
        X_pos = convert_to_rel_pos(X_pos, np.array(self.env.vol.shape[::-1]))
        for i in range(self.obs_buffer["rel_pos"].shape[0]):
            past_pos = self.obs_buffer["rel_pos"][i, 0, :]
            if np.allclose(X_pos, past_pos):
                return True

        return False

    def compute_trajectory(self):
        pass

    def nms_(self, bboxes, scores):
        pass

    def predict_explore_iter_(self, init_pos, if_video=False):
        num_steps = 0
        X_small, X_large, X_pos, X_size = self.env.reset()
        self.env.center = init_pos
        max_ep_len = self.agent.params["max_video_len"]
        (X_small, X_large, X_pos, X_size), _, _, _ = self.env.step((self.env.center, self.env.size))

        if if_video:
            img_clips = []
        bboxes, scores = [], []
        while True:
            num_steps += 1
            action = self.get_action((X_small, X_large, X_pos, X_size))
            (X_small, X_large, X_pos, X_size), _, done, _ = self.env.step(action)
            conf_score = self.classify_()
            if conf_score > self.params["conf_score_threshold"]:
                bboxes.append(center_size2start_end(X_pos, X_size))
                scores.append(conf_score)
            # TODO: comment out
            self.env.render()
            if if_video:
                img_clips.append(self.env.render("rgb_array"))
            if num_steps == max_ep_len:
                done = True
            if done:
                self.clear_buffer_()
                break

        if if_video:
            # TODO: use moviepy to save video
            pass

        return bboxes, scores

    def classify_(self):
        # TODO: use .obs_buffer to predict classification
        conf_score = 1

        return conf_score
