from torch.nn import Identity


encoder_params = {
    "in_channels": 1,
    "patch_size": 16,
    "d_model": 768,
    "nhead": 8,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "num_layers": 12,
    "if_pos_enc": True
}

base_head_params = {
    "in_features": encoder_params["d_model"],
    "out_features": None,
    "hidden_layers": (encoder_params["d_model"], encoder_params["d_model"]),
    "last_act": Identity()
}

patch_pred_head_params = base_head_params.copy()
patch_pred_head_params.update({
    "out_features": encoder_params["d_model"]
})

critic_head_params = base_head_params.copy()
critic_head_params.update({
    "out_features": 1
})

actor_head_params = base_head_params.copy()
actor_head_params.update({
    "out_features": 8  # mu: 3, sigma: 3, cls: 2 (for convenience to convert to Categorical: (pos, neg))
})

# # strictly this should go to configs_ac.py as volumetric_env_params, but for convenience of a param-sharing agent
# vit_agent_params = {
#     "num_target_steps": 10,
#     "num_grad_steps_per_target_update": 10,
#     "lam_cls": 1,
#     "replay_buffer_size": 50000
# }