import subprocess
import argparse


def make_bash_script(hyper_param_dict: dict):
    """
    This script and submitting to sbatch should run in submission/
    Layout:
        + SemesterProject2
        + submission
    cd cmd: start from where the .sh file locates
    """
    bash_script = f"""#!/bin/bash
#SBATCH --output=lesion_detection_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
eval "$(conda shell.bash hook)"
conda activate RL
cd ../SemesterProject2

python scripts/run_vit_greedy_cartesian.py --num_updates {hyper_param_dict["num_updates"]} --if_clip_grad {hyper_param_dict["if_clip_grad"]} --grid_size {hyper_param_dict["grid_size"]}"""

    return bash_script


def create_filename(hyper_param_dict: dict):
    filename = ""
    for key, val in hyper_param_dict.items():
        filename += f"{key}_{val}_"

    filename = filename[:-1].replace(".", "_") + ".sh"

    return filename


if __name__ == '__main__':
    """
    python ./generate_run_vit_greedy_cartesian.py --set_num 1
    """
    hyper_params = dict()
    # set 1
    hyper_params[1] = [
            {"num_updates": 5000, "if_clip_grad": 1, "grid_size": 4},
            {"num_updates": 5000, "if_clip_grad": 1, "grid_size": 5}
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=hyper_params.keys(), required=True)

    args = parser.parse_args()

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
        # print(f"{filename}")
        # print(f"{bash_script}")
        # print("-" * 50)
