import numpy as np
import gym
import moviepy.editor as editor
import os


def Path(obs, image_obs, acts, rewards, next_obs, terminals):
    """
    All inputs: list.
    obs, next_obs: [(N_obs,)...] -> (B, N_obs)
    image_obs: [(H, W, C)...] -> (B, H, W, C) or [] -> (0,)
    acts: [int...] -> (B,) (discrete actions), or [(N_act,)] -> (B, N_act), both: np.array(.)
    rewards: [float...] -> (B,)
    terminals: [bool...] -> (B,)
    """
    if len(image_obs) > 0:
        image_obs = np.stack(image_obs, axis=0)
    return {"observations": np.array(obs),
            "image_obs": np.array(image_obs),
            "actions": np.array(acts),
            "rewards": np.array(rewards),
            "next_obs": np.array(next_obs),
            "terminals": np.array(terminals)}


def print_path(path):
    for key, val in path.items():
        print(f"{key}: {val.shape if isinstance(val, np.ndarray) else len(val)}")


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


def convert_list_of_rollouts(paths):
    """
    Concatenate: [P1["observation"], P2["observation"]...]; list[(B, N_obs)...] -> (P, N_obs)
    """
    obs = np.concatenate([path["observations"] for path in paths], axis=0)  # (P, N_obs)
    actions = np.concatenate([path["actions"] for path in paths], axis=0)  # (P,) or (P, N_act)
    concatenated_rewards = np.concatenate([path["rewards"] for path in paths], axis=0)  # (P,)
    next_obs = np.concatenate([path["next_obs"] for path in paths], axis=0)  # (P, N_obs)
    terminals = np.concatenate([path["terminals"] for path in paths], axis=0)  # (P,)
    unconcatenated_rewards = [path["rewards"] for path in paths]

    return obs, actions, concatenated_rewards, next_obs, terminals, unconcatenated_rewards


def record_gif(frames, save_filename, **kwargs):
    fps = kwargs.get("fps", 25)
    clip = editor.ImageSequenceClip(frames, fps=fps, durations=10)
    clip.write_gif(save_filename, fps)


def create_log_dir_name(timestamp: str, param_dict: dict):
    """
     param_dict should be corrected set up out of the scope. E.g., {"lr": f"{val:.3f}"}
    """
    dirname = timestamp.replace(".", "_")
    for key, val in param_dict.items():
        dirname += f"_{key}_{val}".replace(".", "_")

    # remove the last "_"
    return dirname


def create_param_dir(dirname, pt_filename):
    os.makedirs(dirname)

    return os.path.join(dirname, pt_filename)
