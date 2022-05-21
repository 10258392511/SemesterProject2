import numpy as np
import torch
import torch.nn as nn
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_network as configs_network

from torch.distributions import Normal
from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.modules.vit_agent_modules import EncoderGreedy, MLPHead
from SemesterProject2.helpers.replay_buffer_vit import ReplayBufferGreedy


class ViTGreedyAgent(BaseAgent):
    def __init__(self, params, env):
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
        self.obs_buffer = {"patches_small": None,
                           "patches_large": None,
                           "rel_pos": None}
        self.last_dsc_small = 0
        self.last_dsc_large = 0
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

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
        acc = 0
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
            if X_clf_pred[0, 0] > self.params["conf_score_threshold"] == has_lesion:
                acc += 1
            log_dict["clf_loss"] = loss.item()

        log_dict["clf_acc"] = acc / len(transitions)

        return log_dict

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
