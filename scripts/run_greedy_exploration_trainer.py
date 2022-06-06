import sys

path = "D:\\testings\\Python\\TestingPython"
# path = "/home/zhexwu/Researches/biomedical_imaging"
if path not in sys.path:
    sys.path.append(path)


import argparse
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_network as configs_network
import SemesterProject2.scripts.configs_data_aug as configs_data_aug
import SemesterProject2.helpers.pytorch_utils as ptu

from datetime import datetime as dt
from SemesterProject2.agents.vit_greedy_agent import ViTGreedyAgent
from SemesterProject2.envs.volumetric import VolumetricForGreedy
from SemesterProject2.helpers.vit_greedy_agent_trainer import ViTGreedyAgentTrainer
from SemesterProject2.helpers.utils import create_log_dir_name
from pprint import pprint

if __name__ == '__main__':
    """
    python scripts/run_greedy_exploration_trainer.py --num_episodes 2 --batch_size 2 --print_interval 1 --if_clip_grad --grid_size 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--if_clip_grad", action="store_true")
    parser.add_argument("--grid_size", type=int, default=3)
    args = parser.parse_args()

    train_env_args = {
        "hdf5_filename": configs_data_aug.hdf5_filename,
        "data_splitted_filename": configs_data_aug.data_splitted_filename,
        "mode": "train",
        "seed": None
    }
    train_env_args.update(configs_ac.volumetric_env_params)
    train_env = VolumetricForGreedy(train_env_args)

    test_env_args = {
        "hdf5_filename": configs_data_aug.hdf5_filename,
        "data_splitted_filename": configs_data_aug.data_splitted_filename,
        "mode": "test",
        "seed": None
    }
    test_env_args.update(configs_ac.volumetric_env_params)
    test_env = VolumetricForGreedy(test_env_args)

    vit_agent_args = configs_ac.volumetric_env_params.copy()
    vit_agent_args.update({
        "encoder_params": configs_network.encoder_params,
        "patch_pred_head_params": configs_network.patch_pred_head_params,
        "clf_head_params": configs_network.clf_head_params,
        "encoder_opt_args": configs_network.encoder_opt_args,
        "patch_pred_head_opt_args": configs_network.patch_pred_head_opt_args,
        "clf_head_opt_args": configs_network.clf_head_opt_args
    })
    vit_greedy_agent = ViTGreedyAgent(vit_agent_args, train_env)

    time_stamp = dt.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    vit_trainer_args = vars(args)
    vit_trainer_args["grid_size"] = tuple([vit_trainer_args["grid_size"] for _ in range(3)])
    vit_trainer_args.update({
        "if_notebook": False
    })

    log_dir_params = {
        "if_clip_grad": vit_trainer_args["if_clip_grad"],
        "num_episodes": vit_trainer_args["num_episodes"],
        "batch_size": vit_trainer_args["batch_size"]
    }
    log_dir_name = create_log_dir_name(time_stamp, log_dir_params)
    vit_trainer_args.update({
        "log_dir": f"./run/vit_greedy_exploration/{log_dir_name}",
        "model_save_dir": f"./params/vit_greedy_exploration/{log_dir_name}",
    })
    # pprint(vit_trainer_args)

    # ### VM only ###
    # orig_stdout = sys.stdout
    # orig_stderr = sys.stderr
    # log_file = open(f"/home/zhexwu/Researches/biomedical_imaging/submission/lesion_detection_log/{log_dir_name}.txt",
    #                 "w")
    # sys.stdout = log_file
    # sys.stderr = log_file
    # ### end of VM only block ###

    trainer = ViTGreedyAgentTrainer(vit_trainer_args, vit_greedy_agent, test_env)
    trainer.train(if_record_video=True)
