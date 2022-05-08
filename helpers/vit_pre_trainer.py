import torch
import torch.nn as nn
import SemesterProject2.helpers.pytorch_utils as ptu

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from SemesterProject2.envs.volumetric import Volumetric
from SemesterProject2.agents.policies.sampling_policy import SamplingPolicy
from SemesterProject2.agents.vit_agent import ViTAgent
from SemesterProject2.helpers.replay_buffer_vit import sample_n_trajectories
from SemesterProject2.helpers.utils import create_param_dir, print_list_arrays
from pprint import pprint


class ViTPreTrainer(object):
    def __init__(self, env: Volumetric, sampling_policy: SamplingPolicy, vit_agent: ViTAgent, params):
        """
        params:
        bash:
            num_pre_train_updates, pre_train_batch_size, eval_interval, log_dir, model_save_dir, if_clip_grad, if_notebook
            log_dir: run/volumetric_pre/timestamp, model_dir: params/volumetric_pre/timestamp(/*.pt...)
        """
        self.params = params
        self.agent = vit_agent
        self.env = env
        self.sampling_policy = sampling_policy
        self.encoder_scheduler = ReduceLROnPlateau(self.agent.encoder_opt, factor=0.9, patience=20)
        self.actor_scheduler = ReduceLROnPlateau(self.agent.actor_head_opt, factor=0.9, patience=20)
        self.patch_embed_scheduler = ReduceLROnPlateau(self.agent.patch_pred_head_opt, factor=0.9, patience=20)

        # for pre-training
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}

    def pre_train(self):
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["num_pre_train_updates"], desc="pre_training")

        best_eval_loss = float("inf")
        metric = "eval_loss"
        for num_iter in pbar:
            train_log_dict = self.pre_train_()
            eval_log_dict = self.pre_eval_()

            self.encoder_scheduler.step(eval_log_dict[metric])
            self.actor_scheduler.step(eval_log_dict[metric])
            self.patch_embed_scheduler.step(eval_log_dict[metric])

            if best_eval_loss > eval_log_dict[metric]:
                best_eval_loss = eval_log_dict[metric]
                self.save_models_()

            if num_iter % self.params["eval_interval"] == 0 or num_iter == self.params["num_pre_train_updates"] - 1:
                # logging
                print(f"iter {num_iter} / {self.params['num_pre_train_updates']}")
                ### debugging only ###
                for param in self.agent.encoder_opt.param_groups:
                    print(f"current lr: {param['lr']}")
                    break
                ### end of debugging block ###
                pprint(train_log_dict)
                pprint(eval_log_dict)
                print("-" * 50)

    def pre_train_(self) -> dict:
        """
        tags: train_clf_loss, train_clf_acc, train_novelty_loss, train_loss
        """
        self.agent.encoder.train()
        self.agent.patch_pred_head.train()
        self.agent.actor_head.train()
        if len(self.agent.replay_buffer.paths) < self.params["pre_train_batch_size"]:
            num_paths_to_sample = self.params["pre_train_batch_size"]
        else:
            num_paths_to_sample = 1
        paths = sample_n_trajectories(self.env, self.sampling_policy, num_paths_to_sample,
                                      self.agent.params["max_ep_len"])
        self.agent.add_to_replay_buffer(paths)

        # obs, next_obs: ((T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3), (T, B, 3)), has_seen_lesion: (T, B)
        obs, next_obs, has_seen_lesion = self.agent.replay_buffer.sample_trajetories_eq_len(self.params["pre_train_batch_size"])
        obs, next_obs, has_seen_lesion = self.send_to_device_(obs, next_obs, has_seen_lesion)
        # all (T, B, N_emb)
        embs_encoded, next_embs_encoded, next_embs = self.agent.encoding_(obs, next_obs)
        # (T, B), (1,)
        clf_reward, clf_penalty = self.agent.compute_likelihood_and_penalty_(embs_encoded, None, has_seen_lesion, "clf_only")
        # (1,)
        novelty_loss, _ = self.agent.compute_novelty_seeking_reward_(embs_encoded, next_embs)
        loss = novelty_loss + self.agent.params["lam_cls"] * clf_penalty

        self.agent.encoder_opt.zero_grad()
        self.agent.actor_head_opt.zero_grad()
        self.agent.patch_pred_head_opt.zero_grad()
        loss.backward()
        if self.params["if_clip_grad"]:
            nn.utils.clip_grad_value_(self.agent.encoder.parameters(),
                                      self.agent.params["encoder_opt_args"]["clip_grad_val"])
            nn.utils.clip_grad_value_(self.agent.patch_pred_head.parameters(),
                                      self.agent.params["patch_pred_head_opt_args"]["clip_grad_val"])
            nn.utils.clip_grad_value_(self.agent.actor_head.parameters(),
                                      self.agent.params["actor_head_opt_args"]["clip_grad_val"])
        self.agent.encoder_opt.step()
        self.agent.actor_head_opt.step()
        self.agent.patch_pred_head_opt.step()

        # logging
        clf_acc = clf_reward.mean()
        log_dict = {
            "train_clf_loss": clf_penalty.item(),
            "train_clf_acc": clf_acc.item(),
            "train_novelty_loss": novelty_loss.item(),
            "train_loss": loss.item()
        }

        # print(log_dict)

        for tag, val in log_dict.items():
            self.writer.add_scalar(tag, val, self.global_steps["train"])
        self.global_steps["train"] += 1

        return log_dict

    @torch.no_grad()
    def pre_eval_(self) -> dict:
        """
        tags: eval_clf_loss, eval_clf_acc, eval_novelty_loss, eval_loss
        """
        # obs, next_obs: ((T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3), (T, B, 3)), has_seen_lesion: (T, B)
        self.agent.encoder.eval()
        self.agent.patch_pred_head.eval()
        self.agent.actor_head.eval()
        obs, next_obs, has_seen_lesion = self.agent.replay_buffer.sample_trajetories_eq_len(
            self.params["pre_train_batch_size"])
        obs, next_obs, has_seen_lesion = self.send_to_device_(obs, next_obs, has_seen_lesion)
        # all (T, B, N_emb)
        embs_encoded, next_embs_encoded, next_embs = self.agent.encoding_(obs, next_obs)
        # (T, B), (1,)
        clf_reward, clf_penalty = self.agent.compute_likelihood_and_penalty_(embs_encoded, None, has_seen_lesion,
                                                                             "clf_only")
        # (1,)
        novelty_loss, _ = self.agent.compute_novelty_seeking_reward_(embs_encoded, next_embs)
        loss = novelty_loss + self.agent.params["lam_cls"] * clf_penalty

        # logging
        clf_acc = clf_reward.mean()
        log_dict = {
            "eval_clf_loss": clf_penalty.item(),
            "eval_clf_acc": clf_acc.item(),
            "eval_novelty_loss": novelty_loss.item(),
            "eval_loss": loss.item()
        }

        for tag, val in log_dict.items():
            self.writer.add_scalar(tag, val, self.global_steps["train"])
        self.global_steps["eval"] += 1

        return log_dict

    def save_models_(self):
        torch.save(self.agent.encoder.state_dict(), create_param_dir(self.params["model_save_dir"], "encoder.pt"))
        torch.save(self.agent.actor_head.state_dict(), create_param_dir(self.params["model_save_dir"], "actor_head.pt"))
        torch.save(self.agent.patch_pred_head.state_dict(), create_param_dir(self.params["model_save_dir"], "patch_pred_head.pt"))

    def send_to_device_(self, obs, next_obs, has_seen_lesion):
        obs = [ptu.from_numpy(item).float() for item in obs]
        next_obs = [ptu.from_numpy(item).float() for item in next_obs]
        has_seen_lesion = ptu.from_numpy(has_seen_lesion).float()

        return obs, next_obs, has_seen_lesion
