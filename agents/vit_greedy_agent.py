import numpy as np
import torch
import torch.nn as nn
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_network as configs_network

from torch.distributions import Normal
from sklearn.metrics import precision_score, recall_score
from itertools import product
from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.modules.vit_agent_modules import EncoderGreedy, MLPHead
from SemesterProject2.helpers.replay_buffer_vit import ReplayBufferGreedy
from SemesterProject2.helpers.utils import center_size2start_end, convert_to_rel_pos
from SemesterProject2.envs.volumetric import Volumetric


class ViTGreedyAgent(BaseAgent):
    def __init__(self, params, env: Volumetric):
        """
        params:
        configs_ac.volumetric_env_params:
            num_target_steps, num_grad_steps_per_target_update, lam_cls, replay_buffer_size,
            dice_score_small_th, l2_tao, gamma, if_clip_grad, num_updates_patch_pred, false_neg_weight,
            conf_score_threshold
        As keys: configs_networks:
            encoder_params, *_head_params,
            encoder_opt_args, *_head_opt_args (add clf_head_params & clf_head_opt_args)
        """
        super(ViTGreedyAgent, self).__init__()
        self.env = env
        self.params = params
        self.encoder = EncoderGreedy(self.params["encoder_params"]).to(ptu.device)
        self.patch_pred_head = MLPHead(self.params["patch_pred_head_params"]).to(ptu.device)
        self.clf_head = MLPHead(self.params["clf_head_params"]).to(ptu.device)
        self.replay_buffer = ReplayBufferGreedy(self.params)
        self.encoder_opt = self.params["encoder_opt_args"]["class"](self.encoder.parameters(),
                                                                    **self.params["encoder_opt_args"]["args"])
        self.patch_pred_head_opt = self.params["patch_pred_head_opt_args"]["class"](self.patch_pred_head.parameters(),
                                                                                    **self.params["patch_pred_head_opt_args"]["args"])
        self.clf_head_opt = self.params["clf_head_opt_args"]["class"](self.clf_head.parameters(),
                                                                      **self.params["clf_head_opt_args"]["args"])
        # (T, 1, 1, P, P, P), (T, 1, 1, 2P, 2P, 2P), (T, 1, 3)
        self.obs_buffer = {"patches_small": None,
                           "patches_large": None,
                           "rel_pos": None}
        self.last_dsc_small = 0
        self.last_dsc_large = 0
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.action_space = list(product([-self.params["translation_scale"], 0, self.params["translation_scale"]],
                                         repeat=3))
        self.action_space = [np.array(action) for action in self.action_space if action != (0, 0, 0)]

    def can_sample(self, batch_size):
        return self.replay_buffer.can_sample(batch_size)

    def add_to_replay_buffer(self, X_small, X_large, X_pos, done, info):
        self.replay_buffer.add_to_buffer(X_small, X_large, X_pos, done, info)

    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def train(self, transitions: list):
        transitions = [self.send_to_device_(transition) for transition in transitions]
        log_dict = {}
        patch_pred_log = self.update_patch_pred_head_(transitions)
        clf_log = self.update_clf_head_(transitions)
        log_dict.update(patch_pred_log)
        log_dict.update(clf_log)

        return log_dict

    def update_patch_pred_head_(self, transitions: list):
        print("updating patch pred...")
        log_dict = {}

        for i, transition in enumerate(transitions):
            # print(f"{i + 1}/{len(transitions)}")
            X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion = transition
            # print(f"has_lesion: {has_lesion}")
            X_next_emb = self.encoder.emb_patch(X_small_next, X_large_next).squeeze().detach()  # (1, 1, N_emb) -> (N_emb)
            for _ in range(self.params["num_updates_patch_pred"]):
                X_emb_enc = self.encoder(X_small, X_large, X_pos, X_pos_next)  # (1, N_emb)
                X_patch_pred = self.patch_pred_head(X_emb_enc).squeeze()  # (1, N_emb) -> (N_emb,)
                loss = self.mse(X_emb_enc, X_patch_pred)
                self.encoder_opt.zero_grad()
                self.patch_pred_head_opt.zero_grad()
                loss.backward()
                if self.params["if_clip_grad"]:
                    nn.utils.clip_grad_value_(self.encoder.parameters(),
                                              self.params["encoder_opt_args"]["clip_grad_val"])
                    nn.utils.clip_grad_value_(self.patch_pred_head.parameters(),
                                              self.params["patch_pred_head_opt_args"]["clip_grad_val"])
                self.encoder_opt.step()
                self.patch_pred_head_opt.step()
                log_dict["patch_pred_loss"] = loss.item()

        return log_dict

    def update_clf_head_(self, transitions: list):
        print("updating clf...")
        log_dict = {}
        # acc = 0
        y_pred, y_true = [], []
        for i, transition in enumerate(transitions):
            # print(f"{i + 1}/{len(transitions)}")
            X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion = transition
            # print(f"has_lesion: {has_lesion}")
            X_small = torch.cat([X_small, X_small_next], dim=0)
            X_large = torch.cat([X_large, X_large_next], dim=0)
            X_pos = torch.cat([X_pos, X_pos_next], dim=0)
            X_emb_enc = self.encoder(X_small, X_large, X_pos, X_pos_next)  # (1, N_emb)
            X_clf_pred = self.clf_head(X_emb_enc)  # (1, 2)
            weight = 1 if not has_lesion else self.params["false_neg_weight"]
            print(f"has_lesion: {has_lesion}, weight: {weight}")
            loss = self.ce(X_clf_pred, has_lesion.unsqueeze(0)) * weight
            self.encoder_opt.zero_grad()
            self.clf_head_opt.zero_grad()
            loss.backward()
            if self.params["if_clip_grad"]:
                nn.utils.clip_grad_value_(self.encoder.parameters(),
                                          self.params["encoder_opt_args"]["clip_grad_val"])
                nn.utils.clip_grad_value_(self.clf_head.parameters(),
                                          self.params["clf_head_opt_args"]["clip_grad_val"])
            self.encoder_opt.step()
            self.clf_head_opt.step()
            # if X_clf_pred[0, 1] > self.params["conf_score_threshold"] == has_lesion:
            #     acc += 1
            y_pred.append(ptu.to_numpy(X_clf_pred[0, 1] > self.params["conf_score_threshold"]).astype(int))
            y_true.append(ptu.to_numpy(has_lesion))
            log_dict["clf_loss"] = loss.item()

        # log_dict["clf_acc"] = acc / len(transitions)
        log_dict["clf_acc"] = (y_true == y_pred) / len(transitions)
        log_dict["clf_precision"] = precision_score(y_true, y_pred)
        log_dict["clf_recall"] = recall_score(y_true, y_pred)

        return log_dict

    @torch.no_grad()
    def get_action(self, obs):
        X_small, X_large, X_pos, X_size = obs
        self.add_to_buffer_(X_small, X_large, X_pos)

        next_dsc_small = []
        next_dsc_large = []
        next_novelty = []
        for act in self.action_space:
            X_pos_next_candidate = (X_pos + act * X_size).astype(int)
            # print(X_pos_next_candidate)
            dsc_small = self.compute_dsc_(X_pos_next_candidate, X_size)
            dsc_large = self.compute_dsc_(X_pos_next_candidate, X_size * 2)
            novelty = self.compute_novelty_(X_pos_next_candidate)
            next_dsc_small.append(dsc_small)
            next_dsc_large.append(dsc_large)
            next_novelty.append(novelty)
        next_dsc_small = np.array(next_dsc_small)
        next_dsc_large = np.array(next_dsc_large)
        next_novelty = np.array(next_novelty)
        dsc_improve_small = next_dsc_small - self.last_dsc_small
        dsc_improve_large = next_dsc_large - self.last_dsc_large

        if np.any(dsc_improve_small > self.params["eps_dsc"]):
            # print("chosen from dsc_small")
            ind = np.argmax(dsc_improve_small)
        elif np.any(dsc_improve_large > self.params["eps_dsc"]):
            # print("chosen from dsc_large")
            ind = np.argmax(dsc_improve_large)
        else:
            # print("chosen from novelty")
            # print(next_novelty)
            ind = np.argmax(next_novelty)

        self.last_dsc_small = next_dsc_small[ind]
        self.last_dsc_large = next_dsc_large[ind]
        next_center = X_pos + self.action_space[ind] * X_size
        next_size  = X_size

        return next_center.astype(int), next_size

    def compute_dsc_(self, center, size):
        if not self.env.is_in_vol_(center, size):
            return 0
        start, _ = center_size2start_end(center, size)
        bbox_coord = np.concatenate([start, size], axis=0)  # (6,)
        bbox_mask = self.env.convert_bbox_coord_to_mask_(bbox_coord)
        intersect = (bbox_mask  * self.env.seg).sum()
        dsc = 2 * intersect / (bbox_mask.sum() + self.env.seg.sum())

        return dsc

    def compute_novelty_(self, center):
        center_original = center.copy()
        center = convert_to_rel_pos(center, np.array(self.env.vol.shape[::-1]))
        X_small, X_large, X_pos = self.obs_buffer["patches_small"], self.obs_buffer["patches_large"], \
                                  self.obs_buffer["rel_pos"]
        X_small = ptu.from_numpy(X_small) * 2 - 1
        X_large = ptu.from_numpy(X_large) * 2 - 1
        X_pos = ptu.from_numpy(X_pos)
        X_pos_next = ptu.from_numpy(center).reshape(1, 1, -1)  # (3,) -> (1, 1, 3)
        X_emb_enc = self.encoder(X_small, X_large, X_pos, X_pos_next)  # (1, N_emb)
        X_next_pred = self.patch_pred_head(X_emb_enc).squeeze()  # (1, N_emb) -> (N_emb,)
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

    def clear_buffer(self):
        for key in self.obs_buffer:
            self.obs_buffer[key] = None

    def load_encoder(self, filename):
        assert self.encoder is not None
        self.encoder.load_state_dict(torch.load(filename))
        self.encoder.eval()

    def load_heads(self, patch_pred_filename, clf_filename):
        assert self.patch_pred_head is not None
        assert self.clf_head is not None

        for module_iter, filename_iter in zip((self.patch_pred_head, self.clf_head),
                                              (patch_pred_filename, clf_filename)):
            module_iter.load_state_dict(torch.load(filename_iter))
            module_iter.eval()

    def send_to_device_(self, transition):
        X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion = transition
        X_small = ptu.from_numpy(X_small) * 2 - 1
        X_large = ptu.from_numpy(X_large) * 2 - 1
        X_pos = ptu.from_numpy(X_pos)
        X_small_next = ptu.from_numpy(X_small_next) * 2 - 1
        X_large_next = ptu.from_numpy(X_large_next) * 2 - 1
        X_pos_next = ptu.from_numpy(X_pos_next)
        has_lesion = 1. if has_lesion else 0.
        has_lesion = torch.tensor(has_lesion).to(ptu.device).long()

        return X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion
