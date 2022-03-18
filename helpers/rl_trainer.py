import gym
import numpy as np
import torch
import SemesterProject2.scripts.configs as configs

from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from SemesterProject2.helpers.utils import sample_trajectories, sample_n_trajectories, Path
from SemesterProject2.agents.dqn_agent import DQNAgent
from SemesterProject2.agents.policies.argmax_policy import ArgmaxPolicy


class RLTrainer(object):
    def __init__(self, params):
        """
        params: (agent_params (including opt_params))
            bash script: log_dir, env, if_notebook, ep_len, agent_class, (video_interval) (use print_interval),
            print_interval, batch_size (process from configs (changed to be in configs)), num_agent_train_steps_per_itr,
            eval_batch_size, save_filename, (reset_interval), seed, if_double_q, model_name
        """
        self.params = params
        self.writer = SummaryWriter(log_dir=self.params["log_dir"])
        # self.env = gym.make(self.params["env"])
        # self.eval_env = gym.make(self.params["env"])
        self.env_params = configs.get_env_config(self.params["model_name"])
        self.params.update(self.env_params)
        self.env = configs.get_env(self.params["model_name"])
        self.eval_env = configs.get_env(self.params["model_name"])
        self.agent = self.params["agent_class"](self.params)
        self.total_envsteps = 0
        self.initial_train_average_reward = 0
        self.eps = self.env_params["eps_start"]
        self.env.seed(self.params["seed"])

    def run_training_loop(self, n_itr, collect_policy, eval_policy):
        np.random.seed(self.params["seed"])
        torch.manual_seed(self.params["seed"])

        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(n_itr, desc="training")

        for itr in pbar:
            if_print = (itr % self.params["print_interval"] == 0) or (itr == n_itr - 1)
            if if_print:
                print(f"training: {itr + 1}/{n_itr}", end="")
                print("-" * 50)

            if isinstance(self.agent, DQNAgent):
                all_logs = self.dqn_train_itr()
                pbar.set_description(f"episode train rewards: {all_logs[-1]['train_rewards']:.3f}, "
                                     f"{all_logs[-1]['last_reward']:.3f}")
                paths = []
                video_paths = []

            else:
                paths, timesteps_this_batch, video_paths = self.collect_training_trajectories(collect_policy,
                                                                                              self.env_params["batch_size"],
                                                                                              if_print)
                self.total_envsteps += timesteps_this_batch
                self.agent.add_to_replay_buffer(paths)
                all_logs = self.train_agent()

            if if_print:
                self.perform_logging(itr, paths, eval_policy, video_paths, all_logs, if_print=if_print)
                self.agent.save(self.params["save_filename"])  # TODO: save only the latest model

    def dqn_train_itr(self):
        rewards = 0
        # rewards_list = []
        obs = self.env.reset()
        num_steps = 0
        while True:
            act = self.agent.get_action(obs, eps=self.eps)
            next_obs, reward, done, _ = self.env.step(act)
            rewards += reward
            # rewards_list.append(reward)
            if num_steps > self.params["ep_len"]:
                done = True
            log = self.agent.train(obs, act, reward, next_obs, done)
            obs = next_obs
            self.env.render()
            self.total_envsteps += 1
            # video_paths = []
            num_steps += 1
            if done:
                break

        self.eps = max(self.eps * self.env_params["eps_decay"], self.env_params["eps_end"])

        out_dict = {"train_rewards": rewards, "last_reward": reward}
        out_dict.update(log)
        return [out_dict]

    def collect_training_trajectories(self, collect_policy, num_transitions_to_sample, if_video_path=False):
        paths, timesteps_this_batch = sample_trajectories(self.env, collect_policy, num_transitions_to_sample,
                                                          self.params["ep_len"])
        # sample path for video
        video_paths = []
        if if_video_path:
            video_paths = sample_n_trajectories(self.env, collect_policy, self.env_params["max_num_videos"],
                                                self.env_params["max_video_len"], render=True)

        return paths, timesteps_this_batch, video_paths

    def train_agent(self):
        all_logs = []
        for step in range(self.params["num_agent_train_steps_per_itr"]):
            obs, acts, rewards, next_obs, terminals = self.agent.sample(self.params["batch_size"])
            log = self.agent.train(obs, acts, rewards, next_obs, terminals)
            all_logs.append(log)

        return all_logs

    @torch.no_grad()
    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs, if_print):
        """
        tags: train_video, eval_video, initial_train_average_reward
              train_/eval_: average_reward, std_reward, min_reward, max_reward, average_ep_len
              train_envsteps_so_far
        """
        last_log = all_logs[-1]
        if len(train_video_paths) > 0:
            self._log_video(itr, "train_video", train_video_paths)
        print("Collecting eval video...")
        eval_video_paths = sample_n_trajectories(self.eval_env, eval_policy, self.env_params["max_num_videos"],
                                                 self.env_params["max_video_len"], render=True)
        self.env.close()
        self.eval_env.close()
        # self.env = gym.make(self.params["env"])
        # self.eval_env = gym.make(self.params["env"])

        self._log_video(itr, "eval_video", eval_video_paths)

        print("Collecting eval path...")
        eval_paths, _ = sample_trajectories(self.eval_env, eval_policy, self.params["eval_batch_size"],
                                            self.params["ep_len"])
        train_ep_len = np.array([path["rewards"].shape[0] for path in paths])  # (num_paths,)
        eval_ep_len = np.array([path["rewards"].shape[0] for path in eval_paths])
        train_rewards = np.array([path["rewards"].sum() for path in paths])  # (num_paths,)
        eval_rewards = np.array([path["rewards"].sum() for path in eval_paths])
        # print(f"train_rewards: {train_rewards}")

        if itr == 0:
            self.initial_train_average_reward = np.mean(train_rewards)  # (B,)

        log = OrderedDict()
        if not isinstance(self.agent, DQNAgent):
            log["initial_train_average_reward"] = self.initial_train_average_reward
            log["train_average_reward"] = np.mean(train_rewards)
            log["train_std_reward"] = np.std(train_rewards)
            log["train_min_reward"] = np.min(train_rewards)
            log["train_max_reward"] = np.max(train_rewards)
            log["train_average_ep_len"] = np.mean(train_ep_len)

        log["eval_average_reward"] = np.mean(eval_rewards)
        log["eval_std_reward"] = np.std(eval_rewards)
        log["eval_min_reward"] = np.min(eval_rewards)
        log["eval_max_reward"] = np.max(eval_rewards)
        log["eval_average_ep_len"] = np.mean(eval_ep_len)

        log["train_envsteps_so_far"] = self.total_envsteps
        log.update(last_log)

        for key, val in log.items():
            self.writer.add_scalar(key, val, itr)
            if if_print:
                print(f"{key}: {val:.3f}")
        self.writer.flush()

    def _log_video(self, itr, tag, paths):
        # print(paths[0]["image_obs"].shape)
        img_obs = [path["image_obs"] for path in paths]  # list[(B, H, W, C)]
        # for img_iter in img_obs:
        #     print(img_iter.shape)
        img_obs = np.stack(img_obs, axis=0)  # (N, B, H, W, C)
        img_obs_tensor = torch.tensor(img_obs).permute(0, 1, 4, 2, 3)  # (N, B, C, H, W)
        self.writer.add_video(tag, img_obs_tensor, itr)
