import numpy as np
import torch
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.helpers.pytorch_utils as ptu

from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from SemesterProject2.envs.volumetric import Volumetric
from SemesterProject2.agents.vit_agent import ViTAgent
from SemesterProject2.agents.policies.vit_policy import ViTPolicy
from SemesterProject2.helpers.replay_buffer_vit import sample_trajectories, sample_n_trajectories
from SemesterProject2.helpers.utils import create_param_dir


class ViTAgentTrainer(object):
    def __init__(self, env: Volumetric, agent: ViTAgent, params):
        """
        params: configs.volumetric_env_params
        bash:
            print_interval, log_dir, model_save_dir, if_notebook, num_episodes, pre_train_params_enc_path
        """
        self.params = params
        self.env = env
        self.agent = agent
        agent.load_encoder(self.params["pre_train_params_enc_path"])
        self.policy = ViTPolicy(self.agent, self.env)
        self.best_eval_avg_reward = -float("inf")
        self.inital_train_avg_reward = 0
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_steps = 0

    def run_training_loop(self):
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["num_episodes"], desc="episodes")

        for num_iter in pbar:
            self.global_steps = num_iter
            print("Collecting training path...")
            paths, _ = self.collect_training_trajectories(if_video=False)
            if_print = (num_iter % self.params["print_interval"] == 0 or num_iter == self.params["num_episodes"] - 1)

            self.agent.add_to_replay_buffer(paths)
            train_log_dict = self.train_agent()

            self.perform_logging(paths, train_log_dict, if_print=if_print)

    @torch.no_grad()
    def collect_training_trajectories(self, if_video=False):
        paths, _ = sample_trajectories(self.env, self.policy, self.params["batch_size"], self.params["max_ep_len"])
        video_paths = []
        if if_video:
            video_paths = sample_n_trajectories(self.env, self.policy, self.params["max_num_videos"],
                                                self.params["max_video_len"], render=True)

        return paths, video_paths

    def train_agent(self):
        # only one update
        paths = self.agent.sample(self.params["batch_size"])
        train_log_dict = self.agent.train(paths)

        return train_log_dict

    @torch.no_grad()
    def perform_logging(self, paths, train_log_dict, if_print=False):
        print("Collecting eval paths...")
        eval_paths, eval_video_paths = self.collect_training_trajectories(if_video=False)

        # compute statistics
        train_ep_lens = np.array([path["rewards"].shape[0] for path in paths])
        train_rewards = np.array([path["rewards"].sum() for path in paths])
        eval_ep_lens = np.array([path["rewards"].shape[0] for path in eval_paths])
        eval_rewards = np.array([path["rewards"].sum() for path in eval_paths])
        if self.global_steps == 0:
            self.inital_train_avg_reward = train_rewards.mean()

        log = OrderedDict()
        log["train_avg_reward"] = np.mean(train_rewards)
        log["train_std_reward"] = np.std(train_rewards)
        log["train_min_reward"] = np.min(train_rewards)
        log["train_max_reward"] = np.max(train_rewards)
        log["train_avg_ep_len"] = np.mean(train_ep_lens)

        log["eval_avg_reward"] = np.mean(eval_rewards)
        log["eval_std_reward"] = np.std(eval_rewards)
        log["eval_min_reward"] = np.min(eval_rewards)
        log["eval_max_reward"] = np.max(eval_rewards)
        log["eval_avg_ep_len"] = np.mean(eval_ep_lens)
        log.update(train_log_dict)

        for key, val in log.items():
            self.writer.add_scalar(key, val, self.global_steps)
            if if_print:
                print(f"{key}: {val:.3f}")
        self.writer.flush()
        if if_print:
            print("-" * 100)

        # save the best model
        if log["eval_avg_reward"] > self.best_eval_avg_reward:
            self.best_eval_avg_reward = log["eval_avg_reward"]
            self.save_agent_params()

        # log videos
        if if_print and len(eval_video_paths) > 0:
            # assert len(eval_video_paths) > 0
            eval_img_clips = [path["image_obs"] for path in eval_video_paths]  # list[(T, H, W, 3)]
            eval_img_clips = np.stack(eval_img_clips, axis=0)  # (N, T, H, W, 3)
            eval_img_clips = ptu.from_numpy(eval_img_clips).detach().cpu().permute(0, 1, 4, 2, 3)  # (N, T, 3, H, W)
            self.writer.add_video("eval_video", eval_img_clips, self.global_steps)

    def save_agent_params(self):
        torch.save(self.agent.encoder.state_dict(), create_param_dir(self.params["model_save_dir"], "encoder.pt"))
        torch.save(self.agent.critic_head.state_dict(), create_param_dir(self.params["model_save_dir"], "critic.pt"))
        torch.save(self.agent.actor_head.state_dict(), create_param_dir(self.params["model_save_dir"], "actor.pt"))
        torch.save(self.agent.patch_pred_head.state_dict(), create_param_dir(self.params["model_save_dir"], "patch_pred.pt"))
