import torch
import torch.nn as nn
import random
import SemesterProject2.helpers.pytorch_utils as ptu

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score
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
            num_episodes, batch_size, print_interval, log_dir, model_save_dir, if_clip_grad, if_notebook
            log_dir: run/volumetric_greedy/timestamp, model_dir: params/volumetric_greedy/timestamp(/*.pt...)
        """
        self.params = params
        self.agent = vit_greedy_agent
        self.eval_env = eval_env
        self.eval_replay_buffer = ReplayBufferGreedy(self.params)
        self.eval_policy = FixedSizeSampingPolicy(self.params)
        self.encoder_scheduler = ReduceLROnPlateau(self.agent.encoder_opt, factor=0.9, patience=100)
        self.patch_pred_scheduler = ReduceLROnPlateau(self.agent.patch_pred_head_opt, factor=0.9, patience=100)
        self.clf_scheduler = ReduceLROnPlateau(self.agent.clf_head_opt, factor=0.9, patience=100)

        # for pre-training
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_steps = 0

    def train_(self):
        X_small, X_large, X_pos, X_size = self.agent.env.reset()
        log_dict = {}
        while True:
            act = self.agent.get_action((X_small, X_large, X_pos, X_size))
            (X_small, X_large, X_pos, X_size), _, done, info = self.agent.env.step(act)
            # TODO: comment out
            self.agent.env.render()
            self.agent.add_to_replay_buffer(X_small, X_large, X_pos, done, info)
            if self.agent.can_sample(self.params["batch_size"]):
                transitions = self.agent.sample(self.params["batch_size"])
                log_dict = self.agent.train(transitions)
            if done:
                self.agent.clear_buffer()
                break

        return log_dict

    @torch.no_grad()
    def eval_(self):
        """
        Sample two trajectories: random start and start from a lesion region.
        """
        self.eval_env.reset()
        bbox_coord = random.choice(self.eval_env.bbox_coord)
        bbox_coord_half_len = len(bbox_coord) // 2
        start, size = bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:]
        bbox_center, _ = start_size2center_size(start, size)
        start_pos = [self.eval_env.center, bbox_center]

        for start_pos_iter in start_pos:
            self.eval_env.center = start_pos_iter
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
                y_pred.append(ptu.to_numpy(X_clf_pred[0, 1] > self.params["conf_score_threshold"]).astype(int))
                y_true.append(ptu.to_numpy(has_lesion))

            log_dict["eval_clf_acc"] = (y_true == y_pred) / len(transitions)
            log_dict["eval_clf_precision"] = precision_score(y_true, y_pred)
            log_dict["eval_clf_recall"] = recall_score(y_true, y_pred)
            log_dict["eval_clf_f1"] = f1_score(y_true, y_pred)

        return log_dict

    def train(self):
        if self.params["notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["num_episodes"])

        log_dict = {}
        best_eval_metric_key = "eval_f1_score"
        best_eval_metric = 0
        for i in pbar:
            if_print = (i % self.params["print_interval"] == 0)
            train_log_dict = self.train_()
            eval_log_dict = self.eval_()
            log_dict.update(train_log_dict)
            log_dict.update(eval_log_dict)

            # save models
            if eval_log_dict[best_eval_metric_key] > best_eval_metric:
                best_eval_metric = eval_log_dict[best_eval_metric_key]
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

        for key, val in log_dict:
            self.writer.add_scalar(key, val, self.global_steps)

        if if_print:
            for key, val in log_dict:
                print(f"{key}: {val:.3f}")
            print("-" * 100)
            self.log_video_()

        self.global_steps += 1


    def log_video_(self):
        # collect a path from .eval_env
        pass
