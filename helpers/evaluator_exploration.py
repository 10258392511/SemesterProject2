import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import pickle
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_ac as configs_ac
import SemesterProject2.scripts.configs_data_aug as configs_data_aug
import SemesterProject2.scripts.configs_network as configs_network


from SemesterProject2.helpers.modules.vit_agent_modules import EncoderGreedy, MLPHead
from SemesterProject2.helpers.data_processing import VolumetricDataset
from SemesterProject2.agents.vit_greedy_predictor import ViTGreedyPredictor
from pprint import pprint


def get_param_paths(init_size, params_dir):
    dirname = None
    if init_size == (16, 16, 16):
        dirname = f"{params_dir}/vit_greedy_exploration/diff_init_sizes/2022_06_18_21_39_19_779781_if_clip_grad_True_num_episodes_300_batch_size_2_grid_size_5_init_size_side_16"
    elif init_size == (32, 32, 32):
        dirname = f"{params_dir}/vit_greedy_exploration/diff_init_sizes/2022_06_18_21_39_19_910737_if_clip_grad_True_num_episodes_300_batch_size_2_grid_size_5_init_size_side_32"
    elif init_size == (64, 64, 64):
        dirname = f"{params_dir}/vit_greedy_exploration/diff_init_sizes/2022_06_18_21_39_22_661443_if_clip_grad_True_num_episodes_300_batch_size_2_grid_size_5_init_size_side_64"

    if dirname is None:
        raise IndexError

    param_paths = {
        "encoder": f"{dirname}/encoder.pt",
        "clf_head": f"{dirname}/clf_head.pt",
        "patch_pred_head": f"{dirname}/patch_pred_head.pt"
    }

    return param_paths


def get_translation_scale(init_size):
    if init_size == (16, 16, 16):
        return 0.75

    elif init_size == (32, 32, 32):
        return 0.5

    elif init_size == (64, 64, 64):
        return 0.25

    return 1.


def evaluation_main(param_dir, save_dir, if_notebook=False):
    avg_dict = {
        "train": {},
        "test": {}
    }

    for init_size in [(64, 64, 64), (32, 32, 32), (16, 16, 16)]:
        for threshold in np.arange(0.4, 0.91, 0.1):
            # for mode in ["train", "test"]:
            for mode in ["test"]:
                print(f"current: init_size: {init_size}, th: {threshold}, mode: {mode}")
                # save .csv
                save_dirname_precision = os.path.join(save_dir, f"{mode}/precision/size_{init_size[0]}")
                if not os.path.isdir(save_dirname_precision):
                    os.makedirs(save_dirname_precision)
                save_dirname_recall = os.path.join(save_dir, f"{mode}/recall/size_{init_size[0]}")
                if not os.path.isdir(save_dirname_recall):
                    os.makedirs(save_dirname_recall)
                filename = f"th_{threshold: .1f}".replace(".", "_")
                filename = f"{filename}.csv"
                save_path_precision = os.path.join(save_dirname_precision, filename)
                save_path_recall = os.path.join(save_dirname_recall, filename)

                df_precision, df_recall = evaluation(init_size, threshold, param_dir, mode, if_notebook)
                df_precision.to_csv(save_path_precision)
                df_recall.to_csv(save_path_recall)

                # compute avg
                avg_dict_data = avg_dict[mode]
                avg_dict_data[(threshold, init_size[0])] = {
                    "keys": list(df_precision.keys()),
                    "precision": df_precision.mean().values,
                    "recall": df_recall.mean().values
                }

                # TODO: comment out "print"
                pprint(avg_dict_data)
                print("-" * 100)

                with open(os.path.join(save_dir, "avg_dict.pkl"), "wb") as wf:
                    pickle.dump(avg_dict, wf)


def evaluation(init_size, threshold, param_dir, mode="test", if_notebook=False):
    """
    Returns: df: precision & recall
    """
    assert mode in ("train", "test")
    if if_notebook:
        from tqdm.notebook import trange
    else:
        from tqdm import trange

    ds = VolumetricDataset(configs_data_aug.hdf5_filename, configs_data_aug.data_splitted_filename, mode)
    pbar = trange(len(ds), desc="evaluation")
    log_dict = {
        "precision": {
            "all": []
        },
        "recall": {
            "all": []
        }
    }
    param_path = get_param_paths(init_size, param_dir)

    for index in pbar:
        # TODO: comment out
        print(f"current: {index}")
        # if index != 50:
        #     continue

        video_save_dir = f"./evaluation_exploration/size_{init_size[0]}"
        if not os.path.isdir(video_save_dir):
            os.makedirs(video_save_dir)

        params = configs_ac.volumetric_env_params.copy()
        params.update({
            "hdf5_filename": configs_data_aug.hdf5_filename,
            "data_splitted_filename": configs_data_aug.data_splitted_filename,
            "mode": mode,
            "seed": None,
            "index": index,
            "conf_score_threshold_pred": threshold,
            "encoder_params": configs_network.encoder_params,
            "patch_pred_head_params": configs_network.patch_pred_head_params,
            "clf_head_params": configs_network.clf_head_params,
            "encoder_opt_args": configs_network.encoder_opt_args,
            "patch_pred_head_opt_args": configs_network.patch_pred_head_opt_args,
            "clf_head_opt_args": configs_network.clf_head_opt_args,
            "param_paths": param_path,
            "video_save_dir": video_save_dir,
            "mode_pred": "explore",
            "order": "xyz",
            "grid_size": (2, 2, 5),
            "notebook": if_notebook
        })
        vit_predictor = ViTGreedyPredictor(params)
        bboxes, scores = vit_predictor.predict_explore_(if_video=True)
        selected_bboxes, selected_scores = vit_predictor.nms_(bboxes, scores)
        precision_val, recall_val = vit_predictor.evaluate_faster(selected_bboxes)
        fig = vit_predictor.render_lesion_slices(selected_bboxes)
        fig.savefig(os.path.join(video_save_dir, f"test_{index}", "detection.png"))
        plt.close()


        # params = {
        #     "mode": mode,
        #     "init_size": init_size,
        #     "index": index,
        #     "conf_score_threshold_pred": threshold,
        #     "param_dir": param_dir,
        #     "if_notebook": if_notebook
        # }
        # evaluator = Evaluator(params)
        # log_dict_iter = evaluator.evaluate()
        # TODO: comment out print
        print(f"index {index}: precision: {precision_val}, recall: {recall_val}")

        log_dict["precision"]["all"].append(precision_val)
        log_dict["recall"]["all"].append(recall_val)
        # for stat in log_dict_iter:
        #     assert stat in log_dict
        #     data_dict_iter = log_dict_iter[stat]
        #     data_dict = log_dict[stat]
        #     for direction in data_dict_iter:
        #         assert direction in data_dict
        #         data_dict[direction].append(data_dict_iter[direction])

    df_precision = pd.DataFrame(log_dict["precision"])
    df_recall = pd.DataFrame(log_dict["recall"])
    del vit_predictor

    return df_precision, df_recall
