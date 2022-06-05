import sys

# path = "D:\\testings\\Python\\TestingPython"  # change this after installation (parent dir of the package)
path = "/home/zhexwu/Researches/biomedical_imaging"
if path not in sys.path:
    sys.path.append(path)

import argparse
import SemesterProject2.scripts.configs_data_aug as configs_data_aug
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_network as configs_network

from datetime import datetime as dt
from SemesterProject2.helpers.vit_greedy_cartesian_path_trainer import CartesianTrainer
from SemesterProject2.helpers.utils import create_log_dir_name


if __name__ == '__main__':
    """
    python scripts/run_vit_greedy_cartesian.py --num_updates 5000 --if_clip_grad 0.1 --grid_size 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_updates", type=int, default=5000)
    parser.add_argument("--if_clip_grad", type=float, default=None)
    parser.add_argument("--grid_size", type=int, default=3)
    args = parser.parse_args()
    trainer_params = vars(args)

    time_stamp = dt.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    trainer_params.update({
        "hdf5_filename": configs_data_aug.hdf5_filename,
        "train_test_filename": configs_data_aug.data_splitted_filename,
        "grid_size": tuple([trainer_params["grid_size"] for _ in range(3)]),
        "if_notebook": False
    })
    trainer_params.update(configs_ac.volumetric_env_params)
    trainer_params.update({
        "encoder_params": configs_network.encoder_params,
        "patch_pred_head_params": configs_network.patch_pred_head_params,
        "clf_head_params": configs_network.clf_head_params,
        "encoder_opt_args": configs_network.encoder_opt_args,
        "patch_pred_head_opt_args": configs_network.patch_pred_head_opt_args,
        "clf_head_opt_args": configs_network.clf_head_opt_args
    })

    log_params = {
        "num_updates": trainer_params["num_updates"],
        "if_clip_grad": trainer_params["if_clip_grad"],
        "grid_size": trainer_params["grid_size"][0]
    }

    log_dir_name = create_log_dir_name(time_stamp, log_params)

    trainer_params.update({
        "log_dir": f"./run/vit_greedy_cartesian/{log_dir_name}",
        "params_save_dir": f"./params/vit_greedy_cartesian/{log_dir_name}"
    })

    args = parser.parse_args()
    trainer_params.update(vars(args))
    trainer_params.update({
        "grid_size": tuple([trainer_params["grid_size"] for _ in range(3)])
    })

    ### VM only ###
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(f"/home/zhexwu/Researches/biomedical_imaging/submission/lesion_detection_log/{log_dir_name}.txt",
                    "w")
    sys.stdout = log_file
    sys.stderr = log_file
    ### end of VM only block ###

    cartesian_trainer = CartesianTrainer(trainer_params)
    cartesian_trainer.train()
