import numpy as np
import gym
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_network as configs_network

from monai.transforms import Resize


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
    resize_large = Resize([configs_network.encoder_params["patch_size"] * 2] * 3)

    def convert_obs_to_arrays(obs: list):
        obs_small, obs_large, obs_pos, obs_size = [], [], [], []
        for obs_small_iter, obs_large_iter, obs_pos_iter, obs_size_iter in obs:
            obs_small_iter = resizer_small(obs_small_iter[None, ...])  # (P, P, P) -> (1, P, P, P)
            obs_large_iter = resize_large(obs_large_iter[None, ...])  # (2P, 2P, 2P) -> (1, 2P, 2P, 2P)
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
        if render:
            image_obs.append(env.render(mode="rgb_array"))
            # env.render()
        next_obs.append(ob)
        rews.append(rew)
        terminals.append(bool(done))
        infos.append(info)

        if done:
            policy.terminal = True
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
            replay_buffer_size, dice_small_th, num_steps_to_memorize
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
