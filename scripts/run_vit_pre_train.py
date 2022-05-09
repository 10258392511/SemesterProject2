import sys

# path = "D:\\testings\\Python\\TestingPython"
path = "/home/zhexwu/Researches/biomedical_imaging"
if path not in sys.path:
    sys.path.append(path)

import argparse
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_network as configs_network
import SemesterProject2.scripts.configs_data_aug as configs_data_aug
import SemesterProject2.helpers.pytorch_utils as ptu

from datetime import datetime as dt
from SemesterProject2.agents.vit_agent import ViTAgent
from SemesterProject2.envs.volumetric import Volumetric
from SemesterProject2.agents.policies.sampling_policy import SamplingPolicy
from SemesterProject2.helpers.vit_pre_trainer import ViTPreTrainer
from SemesterProject2.helpers.utils import create_log_dir_name


if __name__ == '__main__':
    """
    python ./scripts/run_vit_pre_train.py
    """
    vm_base_dir = "/itet-stor/zhexwu/net_scratch/semester_project_experiments/deep_rl_lesion_detection"
    local_base_dir = "."

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pre_train_updates", type=int, default=1000)
    parser.add_argument("--pre_train_batch_size", type=int, default=6)
    parser.add_argument("--eval_interval", type=int, default=10)
    trainer_args = vars(parser.parse_args())
    trainer_args.update({
        "if_clip_grad": True,
        "if_notebook": False
    })

    time_stamp = dt.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_dir_params = {
        "num_pre_train_updates": trainer_args["num_pre_train_updates"],
        "pre_train_batch_size": trainer_args["pre_train_batch_size"]
    }
    log_dir_name = create_log_dir_name(time_stamp, log_dir_params)

    ### VM only ###
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(f"/home/zhexwu/Researches/biomedical_imaging/submission/lesion_detection_log/{log_dir_name}.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file
    ### end of VM only block ###

    # TODO: change to tmp/ in VM
    trainer_args.update({
        "log_dir": f"{local_base_dir}/run/vit_pre_train/{log_dir_name}",
        "model_save_dir": f"{local_base_dir}/params/vit_pre_train/{log_dir_name}"
    })

    # Volumetric Env
    env_args = {
        "hdf5_filename": configs_data_aug.hdf5_filename,
        "data_splitted_filename": configs_data_aug.data_splitted_filename,
        "mode": "train",
        "seed": None
    }
    env_args.update(configs_ac.volumetric_env_params)
    volumetric_env = Volumetric(env_args)

    # ViTAgent
    vit_agent_args = configs_ac.volumetric_env_params
    vit_agent_args.update({
        "encoder_params": configs_network.encoder_params,
        "patch_pred_head_params": configs_network.patch_pred_head_params,
        "critic_head_params": configs_network.critic_head_params,
        "actor_head_params": configs_network.actor_head_params,
        "encoder_opt_args": configs_network.encoder_opt_args,
        "patch_pred_head_opt_args": configs_network.patch_pred_head_opt_args,
        "actor_head_opt_args": configs_network.actor_head_opt_args,
        "critic_head_opt_args": configs_network.critic_head_opt_args
    })
    vit_agent = ViTAgent(vit_agent_args)

    # Sampling Policy
    sampling_policy = SamplingPolicy(configs_ac.volumetric_sampling_policy_args)

    # ViTPreTrainer
    vit_pre_trainer = ViTPreTrainer(volumetric_env, sampling_policy, vit_agent, trainer_args)
    vit_pre_trainer.pre_train()
