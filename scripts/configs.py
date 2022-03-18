import torch
import torch.nn as nn
import gym


# Register Env here: 3 functions
def get_network_config(model_name: str):
    if model_name == "LunarLander":
        return lunar_lander_mlp_params, lunar_lander_opt_params

    elif model_name == "LunarLanderImg":
        # TODO
        pass


def get_env_config(model_name: str):
    if model_name == "LunarLander":
        return lunar_lander_env_params

    elif model_name == "LunarLanderImg":
        # TODO
        pass

def get_env(model_name: str):
    if model_name == "LunarLander":
        return gym.make("LunarLander-v2").unwrapped

    elif model_name == "LunarLanderImg":
        # TODO
        pass


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


# max_num_videos = 1
# max_video_len = 600
#
# dqn_eps_params = {
#     "eps_start": 1,
#     "eps_decay": 0.995,
#     "eps_end": 0.01,
#     "tau": 0.001
# }

# LunarLander
lunar_lander_env = gym.make("LunarLander-v2").unwrapped

lunar_lander_max_num_videos = 1
lunar_lander_max_video_len = 600

# lunar_lander_dqn_eps_params = {
#     "eps_start": 1,
#     "eps_decay": 0.995,
#     "eps_end": 0.01,
#     "tau": 0.001
# }

lunar_lander_env_params = base_env_params.copy()
lunar_lander_env_params.update({
    "replay_buffer_size": 50000,
    "batch_size": 64,
    "gamma": 0.999,
    "learning_starts": 300, # 1000
    "update_freq": 1,
    "num_time_steps": 500000,
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
















lunar_lander_conv_params = {
    "model_name": "LunarLanderConv",
    "input_size": (3, 256, 256),
    "act_dim": lunar_lander_env.action_space.n
}






input_shape = (256, 256)

# lunar_lander_env_params["batch_size"] = lunar_lander_env_params["batch_size_obs"] \
#     if not lunar_lander_env_params["image_obs"] else lunar_lander_env_params["batch_size_img_obs"]
#
# lunar_lander_env_params["network_params"] = lunar_lander_mlp_params if not lunar_lander_env_params["image_obs"] \
#     else lunar_lander_conv_params


if __name__ == '__main__':
    pass
