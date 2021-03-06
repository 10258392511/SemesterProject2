# for Actor-Critic
import torch
import torch.nn as nn
import gym


# Register Env here: 3 functions
def get_network_config(model_name: str):
    if model_name == "LunarLander-continuous":
        return lunar_lander_actor_mlp_params, lunar_lander_critic_mlp_params, \
               lunar_lander_actor_opt_params, lunar_lander_critic_opt_params


def get_env_config(model_name: str):
    if model_name == "LunarLander-continuous":
        return lunar_lander_env_params


def get_env(model_name: str):
    if model_name == "LunarLander-continuous":
        return gym.make("LunarLander-v2", continuous=True)


base_env_params = {
    "obs_dim": None,
    "act_dim": None,
    "replay_buffer_size": None,
    "batch_size": None,
    "num_target_updates": None,
    "num_grad_steps_per_target_update": None,
    "num_critic_updates_per_agent_update": None,
    "num_actor_updates_per_agent_update": None,
    "gamma": None,
    "max_num_videos": None,
    "max_video_len": None
}

##########################################################
# LunarLander-v2, continuous
lunar_lander_env_params = base_env_params.copy()
lunar_lander_env_params.update({
    "obs_dim": 8,
    "act_dim": 2,
    "replay_buffer_size": 50000,
    "batch_size": 256,
    "num_target_updates": 10,
    "num_grad_steps_per_target_update": 10,
    "num_critic_updates_per_agent_update": 1,
    "num_actor_updates_per_agent_update": 1,
    "gamma": 0.999,
    "max_num_videos": 1,
    "max_video_len": 600
})

lunar_lander_critic_mlp_params = {
    "obs_dim": 8,
    "act_dim": 1,
    "hidden_layers": [64, 64, 64]
}

lunar_lander_actor_mlp_params = {
    "obs_dim": 8,
    "act_dim": 2,
    "hidden_layers": [64, 64, 64]
}

lunar_lander_critic_opt_params = {
    "constructor": torch.optim.AdamW,
    "optimizer_config": {"lr": 5e-3},
    "loss": nn.MSELoss
}

lunar_lander_actor_opt_params = {
    "constructor": torch.optim.AdamW,
    "optimizer_config": {"lr": 5e-3}
}

##########################################################
# Volumetric
volumetric_env_params = base_env_params.copy()
volumetric_env_params.update({
    "max_ep_len": 30,
    "init_size": (64, 64, 64),  # TODO: only change this; (16, 16, 16)
    "fixed_size": (16, 16, 16),
    "dice_reward_weighting": 100,  # per DQN paper on DCE-MRI, the final bonus
    "fuel_cost": -0.03,
    "dice_score_small_th": 1e-4,
    "eps_dsc": 1e-6,
    "gamma": 0.999,
    "num_target_updates": 2,  # 10
    "num_grad_steps_per_target_update": 2,  # 10
    "num_critic_updates_per_agent_update": 1,  # fixed to be 1
    "num_actor_updates_per_agent_update": 1,  # fixed to be 1
    "lam_cls": 1,
    "replay_buffer_size": 1000,
    "l2_tao": 200,
    "num_steps_to_memorize": 5,
    "max_num_videos": 1,
    "max_video_len": 30,
    "batch_size": 128,
    "if_clip_grad": True,
    "num_updates_patch_pred": 30,
    "num_updates_patch_pred_target": 3,
    "num_updates_clf": 15,
    "false_neg_weight": 1,
    "conf_score_threshold": 0.5,
    "conf_score_threshold_pred": 0.5,
    "pred_zeros_threshold": 1e-2,
    "translation_scale": 3 / 4,
    "iou_threshold": 1e-6,
    "init_perturb_std_ratio": 1  # e.g: 2 * 16
})

if volumetric_env_params["init_size"] == (16, 16, 16):
    volumetric_env_params["translation_scale"] = 0.75
elif volumetric_env_params["init_size"] == (32, 32, 32):
    volumetric_env_params["translation_scale"] = 0.5  # step-size 16
elif volumetric_env_params["init_size"] == (64, 64, 64):
    volumetric_env_params["translation_scale"] = 0.25  # step-size 16

volumetric_sampling_policy_args = {
    "max_ep_len": volumetric_env_params["max_ep_len"],
    "translation_scale": 1 / 3,
    "size_scale": 1 / 6
}

volumetric_sampling_policy_greedy_args = {
    "max_ep_len": volumetric_env_params["max_ep_len"],
    "translation_scale": 1 / 4,
}
