# This file should be run from the root directory

import sys
path = "D:\\testings\\Python\\TestingPython"  # change this after installation (parent dir of the package)
if path not in sys.path:
    sys.path.append(path)

import argparse
import time
import SemesterProject2.scripts.configs as configs

from SemesterProject2.agents.dqn_agent import DQNAgent
from SemesterProject2.agents.policies.argmax_policy import ArgmaxPolicy
from SemesterProject2.helpers.rl_trainer import RLTrainer
from SemesterProject2.helpers.utils import create_log_dir_name, create_param_dir


if __name__ == '__main__':
    """
    python scripts/run_lunar_lander.py --model_name LunarLander --if_double_q --n_itr 500
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--ep_len", type=int, default=600)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--num_agent_train_steps_per_itr", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--if_double_q", action="store_true")
    parser.add_argument("--n_itr", type=int, default=1000)

    args = parser.parse_args()
    params = vars(args)
    params["if_notebook"] = False
    params["agent_class"] = DQNAgent

    params_log = dict(if_double_q=params["if_double_q"],
                      ep_len=params["ep_len"],
                      n_itr=params["n_itr"])

    time_stamp = f"{time.time()}"
    dirname = create_log_dir_name(time_stamp, params_log)
    subdir_name = "dqn_lunar_lander"
    params["log_dir"] = f"run/{subdir_name}/{dirname}"
    params["save_filename"] = create_param_dir(f"params/{subdir_name}/{dirname}", "dqn_agent_q.pt")

    rl_trainer = RLTrainer(params)
    policy = ArgmaxPolicy(rl_trainer.agent)
    rl_trainer.run_training_loop(n_itr=args.n_itr, collect_policy=policy, eval_policy=policy)
