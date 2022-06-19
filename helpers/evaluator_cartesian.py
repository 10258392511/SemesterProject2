import numpy as np
import pandas as pd
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


def get_param_paths(init_size, params_dir):
    dirname = None
    if init_size == (16, 16, 16):
        dirname = f"{params_dir}/vit_greedy_cartesian/diff_sizes/2022_06_15_09_17_12_100361_num_updates_5000_if_clip_grad_1_0_grid_size_7_init_size_side_16"
    elif init_size == (32, 32, 32):
        dirname = f"{params_dir}/vit_greedy_cartesian/diff_sizes/2022_06_15_09_17_12_155409_num_updates_5000_if_clip_grad_1_0_grid_size_7_init_size_side_32"
    elif init_size == (64, 64, 64):
        dirname = f"{params_dir}/vit_greedy_cartesian/diff_sizes/2022_06_15_09_17_12_154825_num_updates_5000_if_clip_grad_1_0_grid_size_7_init_size_side_64"

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

    for init_size in ((16, 16, 16), (32, 32, 32), (64, 64, 64)):
        for threshold in np.arange(0.5, 0.91, 0.1):
            # for mode in ("train", "test"):
            for mode in ("test"):
                print(f"current: init_size: {init_size}, th: {threshold}, mode: {mode}")
                # save .csv
                save_dirname_precision = os.path.join(save_dir, f"{mode}/precision/{init_size[0]}")
                if not os.path.isdir(save_dirname_precision):
                    os.makedirs(save_dirname_precision)
                save_dirname_recall = os.path.join(save_dir, f"{mode}/recall/{init_size[0]}")
                if not os.path.isdir(save_dirname_recall):
                    os.makedirs(save_dirname_recall)
                filename = f"th_{threshold: .1f}.csv"
                save_path_precision = os.path.join(save_dirname_precision, filename)
                save_path_recall = os.path.join(save_dirname_recall, filename)

                df_precision, df_recall = evaluation(init_size, threshold, param_dir, mode, if_notebook)
                df_precision.to_csv(save_path_precision)
                df_recall.to_csv(save_path_recall)

                # compute avg
                avg_dict_data = avg_dict[mode]
                avg_dict_data[threshold] = {
                    "keys": list(df_precision.keys()),
                    "precision": df_precision.mean().values,
                    "recall": df_recall.mean().values
                }

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
            "x": [],
            "y": [],
            "z": [],
            "all": []
        },
        "recall": {
            "x": [],
            "y": [],
            "z": [],
            "all": []
        }
    }
    for index in pbar:
        params = {
            "mode": mode,
            "index": index,
            "conf_score_threshold_pred": threshold,
            "param_dir": param_dir,
            "if_notebook": if_notebook
        }
        evaluator = Evaluator(params)
        log_dict_iter = evaluator.evaluate()
        for stat in log_dict_iter:
            assert stat in log_dict
            data_dict_iter = log_dict_iter[stat]
            data_dict = log_dict[stat]
            for direction in data_dict_iter:
                assert direction in data_dict
                data_dict[direction].append(data_dict_iter[direction])

    df_precision = pd.DataFrame(log_dict["precision"])
    df_recall = pd.DataFrame(log_dict["recall"])

    return df_precision, df_recall


class Evaluator(object):
    """
    Evaluate one vol specified by "mode" and "index"
    """
    def __init__(self, params):
        """
        params: mode, index, conf_score_threshold_pred, init_size, param_dir, if_notebook
        """
        self.params = params
        predictor_args = configs_ac.volumetric_env_params.copy()
        predictor_args.update({
            "encoder_params": configs_network.encoder_params,
            "patch_pred_head_params": configs_network.patch_pred_head_params,
            "clf_head_params": configs_network.clf_head_params,
            "encoder_opt_args": configs_network.encoder_opt_args,
            "patch_pred_head_opt_args": configs_network.patch_pred_head_opt_args,
            "clf_head_opt_args": configs_network.clf_head_opt_args
        })
        predictor_args.update({
            "hdf5_filename": configs_data_aug.hdf5_filename,
            "data_splitted_filename": configs_data_aug.data_splitted_filename,
            "mode": self.params["mode"],
            "seed": None,
            "index": self.params["index"],
            "init_size": self.params["init_size"],
            "translation_scale": get_translation_scale(self.params["init_size"]),
            "conf_score_threshold_pred": self.params["conf_score_threshold_pred"]
        })
        predictor_args.update({
            "video_save_dir": None,  # not used
            "mode_pred": "cartesian",
            "grid_size": (7, 7, 7),
            "param_paths": get_param_paths(self.params["init_size"], self.params["param_dir"]),
            "notebook": self.params["if_notebook"]
        })
        predictor_arg_dict = {
            "x": predictor_args.copy(),
            "y": predictor_args.copy(),
            "z": predictor_args.copy()
        }
        order_dict = {
            "x": "zyx",
            "y": "xzy",
            "z": "yxz"
        }
        for key in predictor_arg_dict:
            predictor_arg_dict[key]["order"] = order_dict[key]

        self.predictors = {}
        for key in predictor_arg_dict:
            predictor_iter = ViTGreedyPredictor(predictor_arg_dict[key])
            predictor_iter.agent.encoder = self.encoder
            predictor_iter.agent.clf_head = self.clf_head
            predictor_iter.agent.patch_pred_head = self.patch_pred_head
            self.predictors[key] = predictor_iter

    def evaluate(self):
        """
        Returns: precision & recall in x, y, z directions and overall.
        {
            precision: {x: float, ..., all: float},
            recall: {x: float, ..., all: float}
        }
        """
        bbox_dict = {}
        score_dict = {}
        log_dict = {
            "precision": {},
            "recall": {}
        }
        for key in self.predictors:
            predictor_iter = self.predictors[key]
            bboxes, scores, _ = predictor_iter.predict_cartesian_()
            selected_bboxes, selected_scores = predictor_iter.nms_(bboxes, scores)
            bbox_dict[key] = selected_bboxes
            score_dict[key] = selected_scores
            precision_val, recall_val = predictor_iter.evaluate_faster(selected_bboxes)
            predictor_iter.trajectory = predictor_iter.init_trajectory_()
            log_dict["precision"][key] = precision_val
            log_dict["recall"][key] = recall_val

        all_selected_bboxes = []
        all_selected_scores = []
        for key in bbox_dict:
            bbox_iter = bbox_dict[key]
            if bbox_iter.ndim == 1:
                bbox_iter = bbox_iter[None, :]
            all_selected_bboxes.append(bbox_iter)
            all_selected_scores.append(score_dict[key])
        all_selected_bboxes = np.concatenate(all_selected_bboxes, axis=0)  # list[(N, 6)] -> (N', 6)
        all_selected_scores = np.concatenate(all_selected_scores, axis=0)  # list[(N,)] -> (N')
        all_selected_bboxes, all_selected_scores = predictor_iter.nms_(all_selected_bboxes, all_selected_scores)
        precision_val, recall_val = predictor_iter.evaluate_faster(all_selected_bboxes)
        log_dict["precision"]["all"] = precision_val
        log_dict["recall"]["all"] = recall_val

        return log_dict
