import sys


# path = "D:\\testings\\Python\\TestingPython"
path = "/home/zhexwu/Researches/biomedical_imaging"
if path not in sys.path:
    sys.path.append(path)


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_data_aug as configs_data_aug

from SemesterProject2.helpers.utils import convert_mpl_to_np, record_gif
from SemesterProject2.agents.policies.sampling_policy import SamplingPolicy
from SemesterProject2.envs.volumetric import Volumetric


if __name__ == '__main__':
    """ 
    Run at root of the project.
    python ./scripts/run_sampling_policy_record.py --if_record
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--if_record", action="store_true")
    parser.add_argument("--gif_save_dir", default="./experiments/volumetric/gifs")
    parser.add_argument("--fps", type=int, default=10)
    args = vars(parser.parse_args())

    env_args = {
        "hdf5_filename": configs_data_aug.hdf5_filename,
        "data_splitted_filename": configs_data_aug.data_splitted_filename
    }
    env_args.update(configs_ac.volumetric_env_params)
    env_args.update(args)

    sampling_policy = SamplingPolicy(configs_ac.volumetric_sampling_policy_args)
    volumetric_env = Volumetric(env_args)

    patch_small, patch_large, center, size = volumetric_env.reset()
    bbox_ind = 2
    volumetric_env.center = np.array([5, 6, 0]) + volumetric_env.bbox_coord[bbox_ind][:3] + volumetric_env.bbox_coord[
                                                                                                bbox_ind][3:]
    volumetric_env.center = (volumetric_env.center).astype(np.int)
    center, size = volumetric_env.center, volumetric_env.size
    ep_len = 100
    img_slices = []
    titles = []
    for i in range(ep_len):
        if env_args["if_record"]:
            volumetric_env.render()
        act_center, act_size = sampling_policy.get_action((patch_small, patch_large, center, size))
        (patch_small, patch_large, center, size), reward, done, _ = volumetric_env.step((act_center, act_size))
        img_slices.append(volumetric_env.render("rgb_array"))
        titles.append(f"step: {i}, rew: {reward:.3f}, done: {done}, center: {center}")
        if done:
            break

    if env_args["if_record"]:
        print("converting mlp images to np images...")
        imgs_out = convert_mpl_to_np(img_slices, titles)
        path_no_title = os.path.join(env_args["gif_save_dir"], "no_title.gif")
        path_title = os.path.join(env_args["gif_save_dir"], "title.gif")
        record_gif(img_slices, path_no_title, fps=env_args["fps"])
        record_gif(imgs_out, path_title, fps=env_args["fps"])

    else:
        for title in titles:
            print(title)
