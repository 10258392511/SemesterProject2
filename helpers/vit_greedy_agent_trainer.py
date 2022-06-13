import numpy as np
import torch
import torch.nn as nn
import random
import SemesterProject2.helpers.pytorch_utils as ptu

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import product, cycle
from SemesterProject2.envs.volumetric import VolumetricForGreedy
from SemesterProject2.helpers.replay_buffer_vit import ReplayBufferGreedy
from SemesterProject2.agents.policies.sampling_policy_greedy import FixedSizeSampingPolicy
from SemesterProject2.agents.vit_greedy_agent import ViTGreedyAgent
from SemesterProject2.helpers.utils import create_param_dir, print_list_arrays, start_size2center_size
from pprint import pprint


class ViTGreedyAgentTrainer(object):
    def __init__(self, params, vit_greedy_agent: ViTGreedyAgent, eval_env: VolumetricForGreedy):
        """
        params:
        bash:
            num_episodes, batch_size, print_interval, grid_size, log_dir, model_save_dir, if_clip_grad, if_notebook
            log_dir: run/volumetric_greedy/timestamp, model_dir: params/volumetric_greedy/timestamp(/*.pt...)
        """
        self.params = params
        self.agent = vit_greedy_agent
        self.eval_env = eval_env
        self.eval_replay_buffer = ReplayBufferGreedy(self.agent.params)
        self.eval_policy = FixedSizeSampingPolicy(self.agent.params)
        self.encoder_scheduler = ReduceLROnPlateau(self.agent.encoder_opt, factor=0.9, patience=100)
        self.patch_pred_scheduler = ReduceLROnPlateau(self.agent.patch_pred_head_opt, factor=0.9, patience=100)
        self.clf_scheduler = ReduceLROnPlateau(self.agent.clf_head_opt, factor=0.9, patience=100)
        # to initialize after env.reset()
        self.init_pos_grid = None  # xyz
        self.vol_shape = None

        # for pre-training
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_train_steps = 0
        self.global_steps = 0

    def init_grid_(self):
        grid_w, grid_h, grid_d = self.params["grid_size"]
        D, H, W = self.vol_shape
        xx = np.linspace(0, W - 1, grid_w + 1)
        yy = np.linspace(0, H - 1, grid_h + 1)
        zz = np.linspace(0, D - 1, grid_d + 1)
        xx = np.clip((xx[:-1] + W / grid_w / 2).astype(int), 0, W - 1)
        yy = np.clip((yy[:-1] + H / grid_h / 2).astype(int), 0, H - 1)
        zz = np.clip((zz[:-1] + D / grid_d / 2).astype(int), 0, D - 1)
        self.init_pos_grid = list(product(xx, yy, zz))

    def train_(self, **kwargs):
        self.agent.encoder.train()
        self.agent.patch_pred_head.train()
        self.agent.clf_head.train()

        if_record_video = kwargs.get("if_record_video", False)
        print("Collecting training transitions...")
        X_small, X_large, X_pos, X_size = self.agent.env.reset()
        self.vol_shape = self.agent.env.vol.shape
        self.init_grid_()

        log_dict = {}
        manual_choice = random.choice([False, True])
        if manual_choice:
            bbox_coord = random.choice(self.agent.env.bbox_coord)
            bbox_coord_half_len = len(bbox_coord) // 2
            start, size = bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:]
            bbox_center, _ = start_size2center_size(start, size)
            self.agent.env.center = (bbox_center + np.random.randn(*X_size.shape) * X_size).astype(int)
            # self.agent.env.center = bbox_center
        else:
            anchor_center = np.array(random.choice(self.init_pos_grid))  # xyz
            print(f"anchor center for normal regions: {anchor_center}")
            center = anchor_center + np.random.randn(*anchor_center.shape) * X_size
            center = center.astype(int)  # xyz
            while not self.agent.env.is_in_vol_(center, 2 * X_size):
                center = anchor_center + np.random.randn(*anchor_center.shape) * X_size
                center = center.astype(int)  # xyz
            self.agent.env.center = center
        (X_small, X_large, X_pos, X_size), _, _, _ = self.agent.env.step((self.agent.env.center, self.agent.env.size))

        if if_record_video:
            train_img_clips = []
        while True:
            act = self.agent.get_action((X_small, X_large, X_pos, X_size))
            (X_small, X_large, X_pos, X_size), _, done, info = self.agent.env.step(act)
            # TODO: comment out
            self.agent.env.render()
            if if_record_video:
                train_img_clips.append(self.agent.env.render("rgb_array"))

            self.agent.add_to_replay_buffer(X_small, X_large, X_pos, done, info)
            if self.agent.can_sample(self.params["batch_size"]):
                transitions = self.agent.sample(self.params["batch_size"])
                print(f"transitions: {[transition[-1] for transition in transitions]}")  # has_lesion
                log_dict = self.agent.train(transitions)

                for key, val in log_dict.items():
                    tag = f"train_{key}"
                    self.writer.add_scalar(tag, val, self.global_train_steps)
                self.global_train_steps += 1

            if done:
                self.agent.clear_buffer()
                break

        if if_record_video:
            log_dict["train_video"] = train_img_clips

        return log_dict

    @torch.no_grad()
    def eval_(self, **kwargs):
        """
        Sample two trajectories: random start and start from a lesion region.
        """
        self.agent.encoder.eval()
        self.agent.patch_pred_head.eval()
        self.agent.clf_head.eval()

        print("Collecting eval transitions...")
        for manual_choice in [False, True]:
            X_small, X_large, X_pos, X_size = self.eval_env.reset()
            self.vol_shape = self.eval_env.vol.shape
            self.init_grid_()
            if manual_choice:
                bbox_coord = random.choice(self.eval_env.bbox_coord)
                bbox_coord_half_len = len(bbox_coord) // 2
                start, size = bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:]
                bbox_center, _ = start_size2center_size(start, size)
                # self.eval_env.center = (bbox_center + np.random.randn(*X_size.shape) * X_size).astype(int)
                self.eval_env.center = bbox_center
            else:
                anchor_center = np.array(random.choice(self.init_pos_grid))  # xyz
                print(f"anchor center for normal regions: {anchor_center}")
                center = anchor_center + np.random.randn(*anchor_center.shape) * X_size
                center = center.astype(int)  # xyz
                while not self.eval_env.is_in_vol_(center, 2 * X_size):
                    center = anchor_center + np.random.randn(*anchor_center.shape) * X_size
                    center = center.astype(int)  # xyz
                self.eval_env.center = center
            (X_small, X_large, X_pos, X_size), _, _, _ = self.eval_env.step((self.eval_env.center, self.eval_env.size))
            while True:
                act = self.eval_policy.get_action((X_small, X_large, X_pos, X_size))
                (X_small, X_large, X_pos, X_size), _, done, info = self.eval_env.step(act)
                # TODO: comment out
                self.eval_env.render()
                self.eval_replay_buffer.add_to_buffer(X_small, X_large, X_pos, done, info)
                if done:
                    break

        log_dict = {}
        if self.eval_replay_buffer.can_sample(self.params["batch_size"]):
            transitions = self.eval_replay_buffer.sample(self.params["batch_size"])
            transitions = [self.agent.send_to_device_(transition) for transition in transitions]
            y_pred, y_true = [], []
            for i, transition in enumerate(transitions):
                # print(f"{i + 1}/{len(transitions)}")
                X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion = transition
                # print(f"has_lesion: {has_lesion}")
                X_small = torch.cat([X_small, X_small_next], dim=0)
                X_large = torch.cat([X_large, X_large_next], dim=0)
                X_pos = torch.cat([X_pos, X_pos_next], dim=0)
                X_emb_enc = self.agent.encoder(X_small, X_large, X_pos, X_pos_next)  # (1, N_emb)
                X_clf_pred = self.agent.clf_head(X_emb_enc)  # (1, 2)
                y_pred.append(ptu.to_numpy(X_clf_pred[0, 1] > self.agent.params["conf_score_threshold"]).astype(int))
                y_true.append(ptu.to_numpy(has_lesion))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            print(f"eval y_true: {y_true}")
            print(f"eval y_pred: {y_pred}")
            log_dict["eval_clf_acc"] = (y_true == y_pred).sum() / len(transitions)
            log_dict["eval_clf_precision"] = precision_score(y_true, y_pred)
            log_dict["eval_clf_recall"] = recall_score(y_true, y_pred)
            log_dict["eval_clf_f1"] = f1_score(y_true, y_pred)

        return log_dict

    def train(self, **kwargs):
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["num_episodes"])

        log_dict = {}
        best_eval_f1, best_eval_acc = 0, 0
        for i in pbar:
            if_print = (i % self.params["print_interval"] == 0)
            train_log_dict = {}
            eval_log_dict = {}
            try:
                train_log_dict = self.train_(**kwargs)
                eval_log_dict = self.eval_(**kwargs)
            except Exception as e:
                print(f"{e}, from .train()")
            # TODO: consider scheduler
            log_dict.update(train_log_dict)
            log_dict.update(eval_log_dict)

            if len(log_dict) == 0:
                continue

            # save models
            # if eval_log_dict["eval_clf_f1"] >= best_eval_f1:
            #     best_eval_f1 = eval_log_dict["eval_clf_f1"]
            #     self.save_models_()
            # elif eval_log_dict["eval_clf_acc"] >= best_eval_acc:
            #     best_eval_acc = eval_log_dict["eval_clf_acc"]
            #     self.save_models_()
            self.save_models_()

            # logging
            self.perform_logging_(log_dict, if_print)

    def save_models_(self):
        torch.save(self.agent.encoder.state_dict(), create_param_dir(self.params["model_save_dir"], "encoder.pt"))
        torch.save(self.agent.patch_pred_head.state_dict(),
                   create_param_dir(self.params["model_save_dir"], "patch_pred_head.pt"))
        torch.save(self.agent.clf_head.state_dict(), create_param_dir(self.params["model_save_dir"], "clf_head.pt"))


    def perform_logging_(self, log_dict: dict, if_print=False):
        # log videos here
        if len(log_dict) == 0:
            return

        for key, val in log_dict.items():
            if "video" not in key:
                self.writer.add_scalar(key, val, self.global_steps)
                print(f"{key}: {val:.3f}")

        if if_print:
            # self.log_video_()
            if "train_video" in log_dict:
                img_clips = log_dict["train_video"]
                img_clips = np.stack(img_clips, axis=0)[None, ...]
                img_clips = torch.tensor(img_clips).permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)
                # print(f"video length: {img_clips.shape}")
                self.writer.add_video("train_video", img_clips, global_step=self.global_steps, fps=15)
        print("-" * 100)

        self.global_steps += 1

    def log_video_(self):
        # collect a path from .eval_env
        print("collecting video...")
        num_steps = 0
        X_small, X_large, X_pos, X_size = self.eval_env.reset()
        max_ep_len = self.agent.params["max_video_len"]
        self.agent.eval_env = self.eval_env
        img_clips = []
        while True:
            num_steps += 1
            action = self.agent.get_action((X_small, X_large, X_pos, X_size))
            (X_small, X_large, X_pos, X_size), _, done, _ = self.eval_env.step(action)
            # TODO: comment out
            self.eval_env.render()
            img_clips.append(self.eval_env.render("rgb_array"))
            if num_steps == max_ep_len:
                done = True
            if done:
                self.agent.clear_buffer()
                break
        # (1, T, H, W, C)
        img_clips = np.stack(img_clips, axis=0)[None, ...]
        img_clips = torch.tensor(img_clips).permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)
        # print(f"video length: {img_clips.shape}")
        self.writer.add_video("eval_video", img_clips, global_step=self.global_steps, fps=15)
        self.agent.eval_env = None
