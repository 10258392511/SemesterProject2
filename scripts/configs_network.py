from torch.nn import Identity
from torch.optim import AdamW

##### model param configs #####
encoder_params = {
    "in_channels": 1,
    "patch_size": 16,
    "d_model": 64,
    "nhead": 4,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "num_layers": 12,
    "if_pos_enc": False
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

##### model optimizers #####
base_opt_args = {
    "class": AdamW,
    "args": {
        "lr": 1e-3
    },
    "clip_grad_val": 1
}

encoder_opt_args = base_opt_args.copy()

patch_pred_head_opt_args = base_opt_args.copy()

critic_head_opt_args = base_opt_args.copy()

actor_head_opt_args = base_opt_args.copy()
