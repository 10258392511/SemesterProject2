import numpy as np
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_network as configs_network

from monai.transforms import Resize


def Paths(obs, image_obs, acts, rewards, next_obs, terminals, infos):
    """
    Parameters
    ----------
    obs: list[((P_x^t, P_y^t, P_z^t), (2P_x^t, 2P_y^t, 2P_z^t), (3,))]
    image_obs: list[(H, W, 3)]
    acts: list[(Nact,)]
    rewards: list[float]
    next_obs: list[((P_x^t, P_y^t, P_z^t), (2P_x^t, 2P_y^t, 2P_z^t), (3,))]
    terminals: list[bool]
    infos: list[dict]: [{key: float}...]

    Returns:
    dict:
        obs: ((T, 1, P, P, P), (T, 1, 2P, 2P, 2P), (T, 3))
        image_obs: (T, H, W, 3)
        acts: (T, Nact)
        rewards: (T,)
        next_obs: ((T, 1, P, P, P), (T, 1, 2P, 2P, 2P), (T, 3))
        terminals: (T,)
        infos: {key: (T,)...}
    """
    resizer_small = Resize([configs_network.encoder_params["patch_size"]] * 3)
    resize_large = Resize([configs_network.encoder_params["patch_size"] * 2] * 3)

    def convert_obs_to_arrays(obs: list):
        obs_small, obs_large, obs_pos = [], [], []
        for obs_small_iter, obs_large_iter, obs_pos_iter in obs:
            obs_small_iter = resizer_small(obs_small_iter)[None, ...]  # (P, P, P) -> (1, P, P, P)
            obs_large_iter = resize_large(obs_large_iter)[None, ...]  # (2P, 2P, 2P) -> (1, 2P, 2P, 2P)
            obs_small.append(obs_small_iter)
            obs_large.append(obs_large_iter)
            obs_pos.append(obs_pos_iter)

        return np.array(obs_small), np.array(obs_large), np.array(obs_pos)

    def convert_info(infos: list):
        out_dict = {}
        for key in infos[0].keys():
            vals = []
            for dict_iter in infos:
                vals.append(dict_iter[key])
            out_dict[key] = np.array(vals)

        return out_dict


    return {
        "observations": convert_obs_to_arrays(obs),
        "image_obs": np.array(image_obs),
        "actions": np.array(acts),
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
    obs, image_obs, acts, next_obs, rews, terminals = [], [], [], [], [], []

    ob = env.reset()
    for _ in range(max_path_length):
        obs.append(ob)
        act = policy.get_action(ob)
        acts.append(act)
        ob, rew, done, _ = env.step(act)
        if render:
            image_obs.append(env.render(mode="rgb_array"))
            # env.render()
        next_obs.append(ob)
        rews.append(rew)
        terminals.append(bool(done))

        if done:
            break

    terminals[-1] = True
    # env.reset()

    return Path(obs, image_obs, acts, rews, next_obs, terminals)


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
        params: replay_buffer_size, dice_small_th
        """
        self.params = params
        self.paths = []
        self.num_transitions = 0

    def add_rollouts(self, paths: dict):
        for path in paths:
            self.paths.append(path)
            self.num_transitions += path["rewards"].shape[0]

    def can_sample(self, batch_size):
        return self.num_transitions >= batch_size

    def sample_recent_rollouts(self, batch_size):
        """
        TODO
        Returns list[Path]
        """
        pass

    def sample_trajetories_eq_len(self, batch_size):
        """
        TODO: randomly choose a path len
        Returns (T, B, 1, P, P, P), (T, B, 1, 2P, 2P, 2P), (T, B, 3)
        """
        pass

    def has_seen_lesion(self, infos: dict):
        """
        infos: list[{"dice_score_small": float, "dice_score_large": float}]
        If dice_score_small > 0: return 1

        Returns
        -------
        labels: (T,)
        """
        return infos["dice_score_small"] > self.params["dice_score_small"]
