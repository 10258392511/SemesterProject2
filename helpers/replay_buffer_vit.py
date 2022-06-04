import numpy as np
import gym
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_network as configs_network

from monai.transforms import Resize
from SemesterProject2.helpers.utils import convert_to_rel_pos


def Path(obs, image_obs, acts, rewards, next_obs, terminals, infos):
    """
    Parameters
    ----------
    obs: list[((P_x^t, P_y^t, P_z^t), (2P_x^t, 2P_y^t, 2P_z^t), (3,), (3,))]
    image_obs: list[(H, W, 3)]
    acts: list[(3,), (3,)]
    rewards: list[float]
    next_obs: list[((P_x^t, P_y^t, P_z^t), (2P_x^t, 2P_y^t, 2P_z^t), (3,))]
    terminals: list[bool]
    infos: list[dict]: [{key: float}...]

    Returns:
    dict:
        obs: ((T, 1, P, P, P), (T, 1, 2P, 2P, 2P), (T, 3), (T, 3))
        image_obs: (T, H, W, 3)
        acts: ((T, 3), (T, 3))
        rewards: (T,)
        next_obs: ((T, 1, P, P, P), (T, 1, 2P, 2P, 2P), (T, 3), (T, 3))
        terminals: (T,)
        infos: {key: (T,)...}
    """
    resizer_small = Resize([configs_network.encoder_params["patch_size"]] * 3)
    resizer_large = Resize([configs_network.encoder_params["patch_size"] * 2] * 3)

    def convert_obs_to_arrays(obs: list):
        obs_small, obs_large, obs_pos, obs_size = [], [], [], []
        for obs_small_iter, obs_large_iter, obs_pos_iter, obs_size_iter in obs:
            obs_small_iter = resizer_small(obs_small_iter[None, ...])  # (P, P, P) -> (1, P, P, P)
            obs_large_iter = resizer_large(obs_large_iter[None, ...])  # (2P, 2P, 2P) -> (1, 2P, 2P, 2P)
            obs_small.append(obs_small_iter)
            obs_large.append(obs_large_iter)
            obs_pos.append(obs_pos_iter)
            obs_size.append(obs_size_iter)

        return np.stack(obs_small, 0), np.stack(obs_large, 0), np.array(obs_pos), np.array(obs_size)

    def convert_act_to_arrays(acts: list):
        act_center, act_size = [], []
        for center_iter, size_iter in acts:
            act_center.append(center_iter)
            act_size.append(size_iter)

        return np.array(act_center), np.array(act_size)

    def convert_info(infos: list):
        out_dict = {}
        for key in infos[0].keys():
            vals = []
            for dict_iter in infos:
                vals.append(dict_iter[key])
            out_dict[key] = np.array(vals)

        return out_dict

    if len(image_obs) > 0:
        image_obs = np.array(image_obs)

    return {
        "observations": convert_obs_to_arrays(obs),
        "image_obs": image_obs,
        "actions": convert_act_to_arrays(acts),
        "rewards": np.array(rewards),
        "next_obs": convert_obs_to_arrays(next_obs),
        "terminals": np.array(terminals),
        "infos": convert_info(infos)
    }


def print_paths(paths: list):
    for path in paths:
        for key, val in path.items():
            print(f"{key}: ", end="")
            if isinstance(val, dict):
                print({key_loc: val_loc.shape for key_loc, val_loc in val.items()})
            elif isinstance(val, tuple) or isinstance(val, list):
                print([item_iter.shape for item_iter in val])
            else:
                print(val.shape)
        print("-" * 50)


def sample_trajectory(env: gym.Env, policy, max_path_length, render=False):
    """
    policy: policy.get_action(obs) -> action
    """
    print("sampling rollout...")
    obs, image_obs, acts, next_obs, rews, terminals, infos = [], [], [], [], [], [], []

    ob = env.reset()
    policy.terminal = False
    for _ in range(max_path_length):
        obs.append(ob)
        act = policy.get_action(ob)
        acts.append(act)
        ob, rew, done, info = env.step(act)
        env.render()
        if render:
            image_obs.append(env.render(mode="rgb_array"))
            # env.render()
        next_obs.append(ob)
        rews.append(rew)
        terminals.append(bool(done))
        infos.append(info)

        if done:
            # print(f"from sample_trajectory(.): done: {done}")
            policy.terminal = True
            policy.clear_buffer()
            break

    terminals[-1] = True
    # env.reset()

    return Path(obs, image_obs, acts, rews, next_obs, terminals, infos)


def sample_trajectories(env: gym.Env, policy, min_timesteps_per_batch, max_path_length, render=False):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path_sampled = sample_trajectory(env, policy, max_path_length, render=render)
        timesteps_this_batch += path_sampled["rewards"].shape[0]  # (B,)
        paths.append(path_sampled)

    return paths, timesteps_this_batch


def sample_n_trajectories(env: gym.Env, policy, n_traj, max_path_length, render=False):
    return [sample_trajectory(env, policy, max_path_length, render=render) for _ in range(n_traj)]


class ReplayBuffer(object):
    def __init__(self, params):
        """
        params: configs_ac.volumetric_env_params
            replay_buffer_size, (dice_small_th, num_steps_to_memorize: to env_params)
        """
        self.params = params
        self.paths = []  # with info being {"has_seen_lesion": (T,)}
        self.num_transitions = 0

    def add_rollouts(self, paths: dict):
        for path in paths:
            path["infos"] = {"has_seen_lesion": self.has_seen_lesion(path["infos"])}
            self.paths.append(path)
            self.num_transitions += path["rewards"].shape[0]
        self.paths = self.paths[-self.params["replay_buffer_size"]:]

    def can_sample(self, batch_size):
        return self.num_transitions >= batch_size

    def sample_recent_rollouts(self, batch_size):
        """
        Returns: list[Path], with "infos" being {"has_seen_legion": (T,)}
        For training: One rollout per iteration.
        """
        num_timesteps = 0
        paths_out = []
        i = -1
        while num_timesteps < batch_size:
            path = self.paths[i]
            paths_out.append(path)
            i -= 1
            num_timesteps += path["rewards"].shape[0]

        return paths_out

    def sample_trajetories_eq_len(self, batch_size, min_path_len=1, max_path_len=30):
        """
        For pre-training: can use batch-training. Returns maximum number of batches (rollouts) in case no sufficient
        rollouts. Considers new rollouts first. Old rollouts are used repeatedly but with varied path length.

        Returns
        -------
        obs, next_obs: ((T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3), (T, B, 3))
            For "patch_pred_head"
        has_seen_lesion: (T, B)
        """
        if_sampled_paths = False
        while not if_sampled_paths:
            path_len = np.random.randint(min_path_len, max_path_len)
            # path_len = max_path_len
            num_rollouts = 0
            obs, next_obs = [[], [], [], []], [[], [], [], []]
            has_seen_lesion = []
            for i in range(len(self.paths) - 1, -1, -1):
                path = self.paths[i]
                if path["rewards"].shape[0] < path_len:
                    continue
                for i_loc in range(len(path["observations"])):
                    # select first path_len timesteps from each rollout
                    obs[i_loc].append(path["observations"][i_loc][:path_len, ...])
                    next_obs[i_loc].append(path["next_obs"][i_loc][:path_len, ...])
                has_seen_lesion.append(path["infos"]["has_seen_lesion"][:path_len, ...])
                num_rollouts += 1
                if num_rollouts == batch_size:
                    break
            if num_rollouts > 0:
                if_sampled_paths = True
            else:
                max_path_len //= 2

        obs = [np.stack(item, axis=1) for item in obs]
        next_obs = [np.stack(item, axis=1) for item in next_obs]
        has_seen_lesion = np.stack(has_seen_lesion, axis=1)

        return obs, next_obs, has_seen_lesion

    def has_seen_lesion(self, infos: dict):
        """
        infos: list[{"dice_score_small": float, "dice_score_large": float}]
        If dice_score_small > 0: return 1

        Returns
        -------
        labels: (T,)
        """
        if_see_lesion = infos["dice_score_small"] > self.params["dice_score_small_th"]
        # has_seen_lesion = (np.cumsum(if_see_lesion.astype(int)) > 0)
        has_seen_lesion =  np.empty(if_see_lesion.shape, dtype=bool)
        for i in range(has_seen_lesion.shape[-1]):
            end = i + 1
            start = max(end - 1 - self.params["num_steps_to_memorize"], 0)
            window = if_see_lesion[start:end]
            has_seen_lesion[i] = np.any(window)

        return has_seen_lesion


class ReplayBufferGreedy(object):
    def __init__(self, params):
        """
        params: configs_ac.volumetric_env_params

        .buffer:
            patches_small: (T, 1, P, P, P),
            patches_large: (T, 1, 2P, 2P, 2P),
            rel_pos: (T, 3),
            has_lesion: (T,), bool
            terminals: (T,), bool
        """
        self.params = params
        self.num_transitions = 0
        self.buffer = {"patches_small": None,
                       "patches_large": None,
                       "rel_pos": None,
                       "has_lesion": None,
                       "terminals": None}
        self.resizer_small = Resize([configs_network.encoder_params["patch_size"]] * 3)
        self.resizer_large = Resize([configs_network.encoder_params["patch_size"] * 2] * 3)

    def __repr__(self):
        out_str = ""
        if self.num_transitions == 0:
            for key, val in self.buffer.items():
                out_str += f"{key}: {val}\n"
        else:
            for key, val in self.buffer.items():
                out_str += f"{key}: {val.shape}\n"

        return out_str

    def can_sample(self, batch_size):
        if batch_size + self.params["num_steps_to_memorize"] > self.num_transitions:
            return False
        return True

    def add_to_buffer(self, X_small, X_large, X_pos, done, info):
        """
        Read "size" from .params.
        All: ndarray
        X_small: (P, P, P), X_large: (2P, 2P, 2P), X_pos: (3,), done: (1,),
        info: {"dice_score_small": float, "dice_score_large": float, "vol_size": ndarray, (3,)}
        """
        self.num_transitions += 1
        X_pos = convert_to_rel_pos(X_pos, info["vol_size"])
        has_lesion = np.array(self.has_lesion_(info)).reshape((1,))  # float -> (1,)
        X_small = self.resizer_small(X_small[None, ...]).squeeze()  # (P, P, P) -> (1, P, P, P) -> (P, P, P)
        X_large = self.resizer_large(X_large[None, ...]).squeeze()
        done = np.array(done).reshape((1,))

        if self.buffer["terminals"] is None:
            self.buffer["patches_small"] = X_small[None, None, ...]
            self.buffer["patches_large"] = X_large[None, None, ...]
            self.buffer["rel_pos"] = X_pos[None, ...]
            self.buffer["terminals"] = done
            self.buffer["has_lesion"] = has_lesion

        else:
            self.buffer["patches_small"] = np.concatenate([self.buffer["patches_small"][-self.params["replay_buffer_size"]:, ...],
                                                           X_small[None, None, ...]], axis=0)
            self.buffer["patches_large"] = np.concatenate([self.buffer["patches_large"][-self.params["replay_buffer_size"]:, ...],
                                                           X_large[None, None, ...]], axis=0)
            self.buffer["rel_pos"] = np.concatenate([self.buffer["rel_pos"][-self.params["replay_buffer_size"]:, ...],
                                                     X_pos[None, ...]], axis=0)
            self.buffer["terminals"] = np.concatenate([self.buffer["terminals"][-self.params["replay_buffer_size"]:, ...],
                                                       done], axis=0)
            self.buffer["has_lesion"] = np.concatenate([self.buffer["has_lesion"][-self.params["replay_buffer_size"]:, ...],
                                                       has_lesion], axis=0)

    def sample(self, batch_size):
        """
        Returns a list of transitions.
        """
        valid_sampling = False
        while not valid_sampling:
            indices = np.random.choice(self.buffer["terminals"].shape[0] - 1, batch_size, replace=False)
            # make sure to sample the most recent one
            indices = np.concatenate([indices, np.array(self.buffer["terminals"].shape[0] - 1 -
                                                        self.params["num_steps_to_memorize"])[None, ...]], axis=0)
            for ind in indices:
                if not self.buffer["terminals"][ind]:
                    valid_sampling = True
                    break

        transitions = []
        for ind in indices:
            if self.buffer["terminals"][ind]:
                continue
            transition = self.sample_transition_(ind)
            transitions.append(transition)

        transitions_lesion = self.sample_transition_lesion_(batch_size)
        transitions += transitions_lesion

        return transitions

    def has_lesion_(self, info: dict):
        return info["dice_score_small"] > self.params["dice_score_small_th"]

    def sample_transition_(self, start_ind):
        """
        Returns:
        X_small: (T, 1, 1, P, P, P),
        X_large: (T, 1, 1, 2P, 2P, 2P),
        X_pos: (T, 1, 3),
        X_small_next: (1, 1, 1, P, P, P),
        X_large_next: (1, 1, 1, 2P, 2P, 2P),
        X_pos_next: (1, 1, 3),
        has_lesion: float
        """
        terminal_window = self.buffer["terminals"][start_ind:start_ind + self.params["num_steps_to_memorize"]]
        end_ind = np.argwhere(terminal_window).flatten()
        # print(f"from .sample_transition_: {end_ind}")
        if len(end_ind) == 0:
            end_ind = min(start_ind + self.params["num_steps_to_memorize"], self.buffer["terminals"].shape[0] - 1)
        else:
            end_ind = start_ind + end_ind[0]

        # print(f"{start_ind}, {end_ind}")

        X_small = self.buffer["patches_small"][start_ind:end_ind, None, ...]  # (t, 1, P, P, P) -> (t, 1, 1, P, P, P)
        X_large = self.buffer["patches_large"][start_ind:end_ind, None, ...]
        X_pos = self.buffer["rel_pos"][start_ind:end_ind, None, ...]
        X_small_next = self.buffer["patches_small"][end_ind:end_ind + 1, None, ...]
        X_large_next = self.buffer["patches_large"][end_ind:end_ind + 1, None, ...]
        X_pos_next = self.buffer["rel_pos"][end_ind:end_ind + 1, None, ...]
        has_lesion = self.buffer["has_lesion"][end_ind]

        return X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion

    def sample_transition_lesion_(self, batch_size):
        # backward: OK to end in a terminal step; whereas the sampling above is forward,
        # i.e. can't start at a terminal step
        transitions = []
        lesion_inds = np.argwhere(self.buffer["has_lesion"]).flatten()
        batch_size = min(batch_size, lesion_inds.shape[0])
        if batch_size == 0:
            return transitions

        lesion_inds_selected = np.random.choice(lesion_inds, batch_size, replace=False)
        for ind in lesion_inds_selected:
            start_ind = max(0, ind - self.params["num_steps_to_memorize"] - 1)
            window = self.buffer["terminals"][start_ind:ind]
            start_ind_offset = np.argwhere(window).flatten()
            if len(start_ind_offset) == 0:
                start_ind_offset = 0
            else:
                start_ind_offset = start_ind_offset[-1]
            start_ind += start_ind_offset
            end_ind = ind

            X_small = self.buffer["patches_small"][start_ind:end_ind, None, ...]  # (t, 1, P, P, P) -> (t, 1, 1, P, P, P)
            X_large = self.buffer["patches_large"][start_ind:end_ind, None, ...]
            X_pos = self.buffer["rel_pos"][start_ind:end_ind, None, ...]
            X_small_next = self.buffer["patches_small"][end_ind:end_ind + 1, None, ...]
            X_large_next = self.buffer["patches_large"][end_ind:end_ind + 1, None, ...]
            X_pos_next = self.buffer["rel_pos"][end_ind:end_ind + 1, None, ...]
            has_lesion = self.buffer["has_lesion"][end_ind]

            transitions.append((X_small, X_large, X_pos, X_small_next, X_large_next, X_pos_next, has_lesion))

        return transitions

    @staticmethod
    def print_transitions(transitions):
        for transition in transitions:
            print([item.shape for item in transition])
            print("-" * 100)
