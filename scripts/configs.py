import torch
import torch.nn as nn
import gym


# Register Env here: 3 functions
def get_network_config(model_name: str):
    if model_name == "LunarLander":
        return lunar_lander_mlp_params, lunar_lander_opt_params

    elif model_name == "LunarLanderImg":
        return lunar_lander_img_conv_params, lunar_lander_img_opt_params


def get_env_config(model_name: str):
    if model_name == "LunarLander":
        return lunar_lander_env_params

    elif model_name == "LunarLanderImg":
        return lunar_lander_img_conv_params

def get_env(model_name: str):
    env = None
    if model_name == "LunarLander":
        env = gym.make("LunarLander-v2").unwrapped

    elif model_name == "LunarLanderImg":
        env = LunarLanderImg()

    return env


base_env_params = {
    "replay_buffer_size": None,
    "batch_size": None,
    "update_freq": None,
    "gamma": None,
    "tau": None,
    "eps_start": None,
    "eps_decay": None,
    "eps_end": None,
    "max_num_videos": None,
    "max_video_len": None
}

##########################################################
# LunarLander
lunar_lander_env = gym.make("LunarLander-v2").unwrapped

lunar_lander_env_params = base_env_params.copy()
lunar_lander_env_params.update({
    "replay_buffer_size": 50000,
    "batch_size": 64,
    "gamma": 0.999,
    "update_freq": 1,
    "eps_start": 1,
    "eps_decay": 0.995,
    "eps_end": 0.01,
    "tau": 0.001,
    "max_num_videos": 1,
    "max_video_len": 600
})

lunar_lander_opt_params = {
    "constructor": torch.optim.Adam,
    "optimizer_config": {"lr": 1e-3},
    "loss": nn.SmoothL1Loss
}

lunar_lander_mlp_params = {
    "obs_dim": lunar_lander_env.observation_space.shape[0],
    "act_dim": lunar_lander_env.action_space.n,
    "hidden_layers": [64, 64]
}


##########################################################
# LunarLanderImg
# lunar_lander_img_env = lunar_lander_img.LunarLanderImg()

lunar_lander_img_env_params = base_env_params.copy()
lunar_lander_img_env_params.update({
    "replay_buffer_size": 50000,
    "batch_size": 8,
    "gamma": 0.999,
    "update_freq": 1,
    "eps_start": 1,
    "eps_decay": 0.995,
    "eps_end": 0.01,
    "tau": 0.001,
    "max_num_videos": 1,
    "max_video_len": 600,
    "input_shape": (128, 128)
})

lunar_lander_img_conv_params = {
    "model_name": "LunarLanderConv",
    "input_size": (3, 256, 256),
    "act_dim": 4
}

lunar_lander_img_opt_params = {
    "constructor": torch.optim.Adam,
    "optimizer_config": {"lr": 1e-3},
    "loss": nn.SmoothL1Loss
}


if __name__ == '__main__':
    pass
