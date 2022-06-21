import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn as nn
import os
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_network as configs_network

from monai.transforms import Resize
from itertools import product
from SemesterProject2.agents.vit_greedy_agent import ViTGreedyAgent
from SemesterProject2.envs.volumetric import VolumetricForGreedy
from SemesterProject2.helpers.utils import center_size2start_end, start_size2center_size, start_size2start_end, \
    create_param_dir, convert_to_rel_pos, record_gif


class ViTGreedyPredictor(object):
    def __init__(self, params):
        """
        params: configs_ac.volumetric_env_params, configs_network.*_params
        bash:
            param_paths: {"encoder": "*.pt", "clf_head": ..., "patch_pred_head": ...}
            video_save_dir
            mode_pred: "cartesion" or "explore"
            order: "zyx"...
            grid_size: (3, 3, 3), "xyz"
            notebook
        """
        self.params = params
        self.env = VolumetricForGreedy(params)
        self.env.reset()
        self.agent = ViTGreedyAgent(params, self.env)
        if self.params["param_paths"]["encoder"] is not None:
            self.agent.load_encoder(self.params["param_paths"]["encoder"])
        if self.params["param_paths"]["patch_pred_head"] is not None and \
            self.params["param_paths"]["clf_head"] is not None:
            self.agent.load_heads(self.params["param_paths"]["patch_pred_head"],
                                  self.params["param_paths"]["clf_head"])
        if self.params["mode_pred"] != "explore":
            self.ordering = {"z": 0, "y": 1, "x": 2}
            self.trajectory = self.init_trajectory_()  # [(X_pos, terminal)], generator
            step_size = np.array(self.params["init_size"]) * self.params["translation_scale"]
            self.trajectory_len = int(np.product(self.env.vol.shape) / np.product(step_size))
        # (T, 1, 1, P, P, P), (T, 1, 2P, 2P, 2P), (T, 1, 3)
        self.obs_buffer = {
            "patches_small": None,
            "patches_large": None,
            "rel_pos": None
        }
        if self.params["mode_pred"] == "explore":
            self.init_pos_grid = None  # (N, 3)
            self.init_grid_()
        self.mse = nn.MSELoss()
        self.action_space = list(product([-self.params["translation_scale"], 0, self.params["translation_scale"]],
                                         repeat=3))
        self.action_space = [np.array(action) for action in self.action_space if action != (0, 0, 0)]
        self.resizer_small = Resize([configs_network.encoder_params["patch_size"]] * 3, mode="trilinear")
        self.resizer_large = Resize([configs_network.encoder_params["patch_size"] * 2] * 3, mode="trilinear")

    @torch.no_grad()
    def predict(self, **kwargs):
        """
        Returns: bboxes: (N, 6), [start, end]; conf_scores: (N,)
        """
        if self.params["mode_pred"] == "explore":
            assert self.init_pos_grid is not None
            if_video = kwargs.get("if_video", False)

            return self.predict_explore_(if_video)

        assert self.trajectory is not None
        if self.params["mode_pred"] == "cartesian":
            return self.predict_cartesian_()

    @torch.no_grad()
    def evaluate(self, bboxes):
        def evaluate_iter(bbox):
            patch_mask = self.env.convert_bbox_coord_to_mask_(bbox)
            intersection = (self.env.seg * patch_mask).sum()
            dice_score = 2 * intersection / (patch_mask.sum() + self.env.seg.sum())

            return dice_score

        dice_scores = []
        for bbox in bboxes:
            dice_scores.append(evaluate_iter(bbox))
        dice_scores = np.array(dice_scores)

        return dice_scores, dice_scores > self.params["dice_score_small_th"]

    @torch.no_grad()
    def evaluate_faster(self, bboxes):
        if len(bboxes) == 0:
            return 0., 0.
        iou_matrix = np.zeros((len(self.env.bbox_coord), len(bboxes)))  # (N_gt_boxes, N_pred_boxes)
        for i, bbox_gt in enumerate(self.env.bbox_coord):
            half_len = len(bbox_gt) // 2
            bbox_gt_start, bbox_gt_end = start_size2start_end(bbox_gt[:half_len], bbox_gt[half_len:])
            bbox_gt = np.concatenate([bbox_gt_start, bbox_gt_end], axis=-1)
            for j, bbox in enumerate(bboxes):
                iou_matrix[i, j] = self.iou_(bbox_gt, bbox)

        iou_matrix_th = (iou_matrix > self.params["iou_threshold"])
        recall_val = np.max(iou_matrix_th, axis=1).sum() / iou_matrix_th.shape[0]
        precision_val = np.max(iou_matrix_th, axis=0).sum() / iou_matrix_th.shape[1]

        return precision_val, recall_val

    @torch.no_grad()
    def render_lesion_slices(self, bboxes):
        # renders the slices at center of a GT lesion bbox
        def render_lesion_slice(slice_ind):
            img_slice = (self.env.vol[slice_ind, ...] * 255).astype(np.uint8)
            img_slice = cv.cvtColor(img_slice, cv.COLOR_GRAY2RGB)
            seg_slice_mask = (self.env.seg[slice_ind, ...] > 0)
            # red
            img_slice[seg_slice_mask, 0] = 255
            img_slice[seg_slice_mask, 1:] = 0

            # GT: red
            for bbox_coord in self.env.bbox_coord:
                # (x_min, y_min, z_min, x_size, y_size, z_size)
                bbox_coord_half_len = len(bbox_coord) // 2
                bbox_start, bbox_end = start_size2start_end(bbox_coord[:bbox_coord_half_len],
                                                            bbox_coord[bbox_coord_half_len:])
                if bbox_start[-1] <= slice_ind <= bbox_end[-1]:
                    # red
                    cv.rectangle(img_slice, bbox_start[:2], bbox_end[:2], color=(255, 0, 0), thickness=2,
                                 lineType=cv.LINE_AA)
            # predictions: blue
            for bbox_coord in bboxes:
                # (x_min, y_min, z_min, x_size, y_size, z_size)
                bbox_coord_half_len = len(bbox_coord) // 2
                bbox_start, bbox_end = bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:]
                if bbox_start[-1] <= slice_ind <= bbox_end[-1]:
                    # blue
                    cv.rectangle(img_slice, bbox_start[:2], bbox_end[:2], color=(0, 0, 255), thickness=2,
                                 lineType=cv.LINE_AA)

            return img_slice

        num_bboxes = len(self.env.bbox_coord)
        num_cols = 3
        num_rows = np.ceil(num_bboxes / num_cols).astype(int)
        num_rows = 100
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10.8, 3.6 * num_rows))
        axes_flatten = axes.flatten()

        # for i, bbox_coord in enumerate(self.env.bbox_coord):
        #     bbox_coord_half_len = len(bbox_coord) // 2
        #     bbox_center, _ = start_size2center_size(bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:])
        #     slice_ind = bbox_center[-1]
        #     img_slice = render_lesion_slice(slice_ind)
        #     axis = axes_flatten[i]
        #     axis.imshow(img_slice)
        #     axis.set_title(f"z = {slice_ind}")

        i = -1
        for bbox_coord in self.env.bbox_coord:
            for slice_ind in range(bbox_coord[2], bbox_coord[2] + bbox_coord[5] + 1):
                i += 1
                img_slice = render_lesion_slice(slice_ind)
                axis = axes_flatten[i]
                axis.imshow(img_slice)
                axis.set_title(f"z = {slice_ind}")

        for j in range(i + 1, axes_flatten.shape[0]):
            fig.delaxes(axes_flatten[j])

        fig.tight_layout()
        return fig

    @torch.no_grad()
    def get_action(self, obs):
        X_small, X_large, X_pos, X_size = obs
        self.add_to_buffer_(X_small, X_large, X_pos)

        next_novelty = []
        for act in self.action_space:
            X_pos_next_candidate = (X_pos + act * X_size).astype(int)
            # print(X_pos_next_candidate)
            if (not self.env.is_in_vol_(X_pos_next_candidate, X_size * 2)) or \
                    self.recently_visited_(X_pos_next_candidate):
                novelty = 0
            else:
                novelty = self.compute_novelty_(X_pos_next_candidate)
            next_novelty.append(novelty)
        next_novelty = np.array(next_novelty)
        ind = np.argmax(next_novelty)

        next_center = X_pos + self.action_space[ind] * X_size
        next_size = X_size

        return next_center.astype(int), next_size

    def init_trajectory_(self):
        # e.g. xzy: 2, 0, 1 -> xyz: 2, 1, 0: P = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
        ordering_in_inds = [self.ordering[dim] for dim in self.params["order"]]
        ordering_out_inds = [self.ordering[dim] for dim in "xyz"]
        permute_mat = np.zeros((3, 3))
        for i, ind_out in enumerate(ordering_out_inds):
            for j, ind_in in enumerate(ordering_in_inds):
                if ind_in == ind_out:
                    permute_mat[i, j] = 1.

        shape = np.array(self.env.vol.shape)[ordering_in_inds]
        step_size = np.array(self.params["init_size"])[ordering_in_inds] * self.params["translation_scale"]
        step_size = step_size.astype(int)
        for i_iter in range(0, shape[0], step_size[0]):
            for j_iter in range(0, shape[1], step_size[1]):
                for k_iter in range(0, shape[2], step_size[2]):
                    # e.g. xzy -> xyz
                    coord = permute_mat @ np.array([i_iter, j_iter, k_iter])
                    terminal = False
                    if k_iter + step_size[2] > shape[2] - 1:
                        terminal = True
                    yield coord, terminal

    def init_grid_(self):
        grid_w, grid_h, grid_d = self.params["grid_size"]
        D, H, W = self.env.vol.shape
        xx = np.linspace(0, W - 1, grid_w + 1)
        yy = np.linspace(0, H - 1, grid_h + 1)
        zz = np.linspace(0, D - 1, grid_d + 1)
        xx = np.clip((xx[:-1] + W / grid_w / 2).astype(int), 0, W - 1)
        yy = np.clip((yy[:-1] + H / grid_h / 2).astype(int), 0, H - 1)
        zz = np.clip((zz[:-1] + D / grid_d / 2).astype(int), 0, D - 1)
        self.init_pos_grid = np.array(list(product(xx, yy, zz)))

    def add_to_buffer_(self, X_small, X_large, X_pos):
        """
        All: ndarray
        X_small: (P, P, P), X_large: (2P, 2P, 2P), X_pos: (3,)
        """
        X_pos = convert_to_rel_pos(X_pos, np.array(self.env.vol.shape[::-1]))
        # TODO: resize
        X_small = self.resizer_small(X_small[None, ...])  # (1, P, P, P)
        X_large = self.resizer_large(X_large[None, ...])  # (1, 2P, 2P, 2P)
        X_small = X_small[None, None, ...]  # (1, 1, 1, P, P, P)
        X_large = X_large[None, None, ...]  # (1, 1, 1, 2P, 2P, 2P)
        X_pos = X_pos[None, None, ...]  # (1, 1, 3)
        if self.obs_buffer["rel_pos"] is None:
            self.obs_buffer["patches_small"] = X_small
            self.obs_buffer["patches_large"] = X_large
            self.obs_buffer["rel_pos"] = X_pos
        else:
            if self.obs_buffer["rel_pos"].shape[0] < self.params["num_steps_to_memorize"]:
                self.obs_buffer["patches_small"] = np.concatenate([self.obs_buffer["patches_small"], X_small], axis=0)
                self.obs_buffer["patches_large"] = np.concatenate([self.obs_buffer["patches_large"], X_large], axis=0)
                self.obs_buffer["rel_pos"] = np.concatenate([self.obs_buffer["rel_pos"], X_pos], axis=0)
            else:
                self.obs_buffer["patches_small"] = np.concatenate([self.obs_buffer["patches_small"][1:, ...], X_small],
                                                                  axis=0)
                self.obs_buffer["patches_large"] = np.concatenate([self.obs_buffer["patches_large"][1:, ...], X_large],
                                                                  axis=0)
                self.obs_buffer["rel_pos"] = np.concatenate([self.obs_buffer["rel_pos"][1:, ...], X_pos], axis=0)

    def clear_buffer_(self):
        for key in self.obs_buffer:
            self.obs_buffer[key] = None

    def compute_novelty_(self, center):
        center_original = center.copy()
        center = convert_to_rel_pos(center, np.array(self.env.vol.shape[::-1]))
        X_small, X_large, X_pos = self.obs_buffer["patches_small"], self.obs_buffer["patches_large"], \
                                  self.obs_buffer["rel_pos"]
        X_small = ptu.from_numpy(X_small) * 2 - 1
        X_large = ptu.from_numpy(X_large) * 2 - 1
        X_pos = ptu.from_numpy(X_pos)
        X_pos_next = ptu.from_numpy(center).reshape(1, 1, -1)  # (3,) -> (1, 1, 3)
        X_emb_enc = self.agent.encoder(X_small, X_large, X_pos, X_pos_next)  # (1, N_emb)
        X_next_pred = self.agent.patch_pred_head(X_emb_enc).squeeze()  # (1, N_emb) -> (N_emb,)
        # (P, P, P)
        X_next_candidate_small = self.env.get_patch_by_center_size(center_original,
                                                                   np.array(self.obs_buffer["patches_small"].shape[3:]))
        # (2P, 2P, 2P)
        X_next_candidate_large = self.env.get_patch_by_center_size(center_original,
                                                                   np.array(self.obs_buffer["patches_large"].shape[3:]))
        # (1, 1, 1, P, P, P), (1, 1, 1, 2P, 2P, 2P)
        X_next_candidate_small = ptu.from_numpy(X_next_candidate_small).reshape(1, 1, 1, *X_next_candidate_small.shape) * 2 - 1
        X_next_candidate_large = ptu.from_numpy(X_next_candidate_large).reshape(1, 1, 1, *X_next_candidate_large.shape) * 2 - 1
        X_next_patch_emb = self.agent.encoder.emb_patch(X_next_candidate_small, X_next_candidate_large).squeeze()  # (1, 1, N_emb) -> (N_emb)
        novelty = self.mse(X_next_pred, X_next_patch_emb)

        return novelty.item()

    def recently_visited_(self, X_pos):
        X_pos = convert_to_rel_pos(X_pos, np.array(self.env.vol.shape[::-1]))
        for i in range(self.obs_buffer["rel_pos"].shape[0]):
            past_pos = self.obs_buffer["rel_pos"][i, 0, :]
            if np.allclose(X_pos, past_pos):
                return True

        return False

    def nms_(self, bboxes, scores):
        # time complexity: O(n)
        indices = np.argsort(scores, axis=-1)
        indices = indices[::-1]
        scores = scores[indices]
        bboxes = bboxes[indices, ...]
        visited = np.zeros_like(scores, dtype=bool)
        selected_bboxes = []
        selected_scores = []

        for i in range(scores.shape[0]):
            if visited[i]:
                continue
            visited[i] = True
            bbox_iter = bboxes[i, ...]
            selected_bboxes.append(bbox_iter)
            selected_scores.append(scores[i])
            for j in range(i + 1, scores.shape[0]):
                if visited[j]:
                    continue
                bbox_other_iter = bboxes[j, ...]
                iou_score = self.iou_(bbox_iter, bbox_other_iter)
                if iou_score > self.params["iou_threshold"]:
                    visited[j] = True

        return np.array(selected_bboxes), np.array(selected_scores)

    def iou_(self, bbox1, bbox2):
        # bbox1, bbox2: (6,): start, end
        def compute_vol(start, end):
            sides = np.maximum(end - start, 0)

            return np.product(sides)

        bbox_half_len = len(bbox1) // 2
        top_left = np.stack([bbox1[:bbox_half_len], bbox2[:bbox_half_len]], axis=0)  # (2, 3)
        bottom_right = np.stack([bbox1[bbox_half_len:], bbox2[bbox_half_len:]], axis=0)  # (2, 3)
        intersec_top_left = np.max(top_left, axis=0)  # (3,)
        intersec_bottom_right = np.min(bottom_right, axis=0)  # (3,)
        intersec_area = compute_vol(intersec_top_left, intersec_bottom_right)
        area1 = compute_vol(bbox1[:bbox_half_len], bbox1[bbox_half_len:])
        area2 = compute_vol(bbox2[:bbox_half_len], bbox2[bbox_half_len:])
        union_area = area1 + area2 - intersec_area
        # print(f"area1: {area1}, area2: {area2}, union: {union_area}, intersection: {intersec_area}")

        return intersec_area / union_area

    def predict_cartesian_(self):
        bboxes, scores, all_scores = [], [], []
        X_size = np.array(self.params["init_size"])
        if self.params["notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.trajectory, total=self.trajectory_len, desc="testing: cartesian")
        for X_pos, terminal in pbar:
            if not self.env.is_in_vol_(X_pos, 2 * X_size):
                self.clear_buffer_()
                continue
            X_pos_orig = X_pos.copy()
            X_patch_small = self.env.get_patch_by_center_size(X_pos, X_size)
            if np.all(np.abs(X_patch_small) < self.params["pred_zeros_threshold"]):
                # print(f"zero region: {X_pos}")
                self.clear_buffer_()
                continue
            X_patch_large = self.env.get_patch_by_center_size(X_pos, 2 * X_size)
            self.add_to_buffer_(X_patch_small, X_patch_large, X_pos)

            X_small = self.obs_buffer["patches_small"]
            X_large = self.obs_buffer["patches_large"]
            X_pos = self.obs_buffer["rel_pos"]
            X_small = ptu.from_numpy(X_small) * 2 - 1  # (T, 1, 1, P, P, P)
            X_large = ptu.from_numpy(X_large) * 2 - 1  # (T, 1, 1, 2P, 2P, 2P)
            X_pos = ptu.from_numpy(X_pos)  # (T, 1, 3)
            X_enc = self.agent.encoder(X_small, X_large, X_pos, X_pos[-1:, ...])  # (1, N_emb)
            X_clf_pred = self.agent.clf_head(X_enc)  # (1, 2)
            X_clf_pred = torch.softmax(X_clf_pred, dim=-1)
            score_pred = X_clf_pred[0, 1].item()
            all_scores.append(score_pred)
            if score_pred > self.params["conf_score_threshold_pred"]:
            # if score_pred > 0.5:
                start, end = center_size2start_end(X_pos_orig, X_size)
                bboxes.append(np.concatenate([start, end], axis=-1))
                scores.append(score_pred)

                ### debug only ###
                # print(f"current pos: {X_pos_orig}, score: {score_pred}")
                # pbar.set_description(f"current pos: {X_pos_orig}, score: {score_pred}")
                ### end of debug block ###
            pbar.set_description(f"current pos: {X_pos_orig}, score: {score_pred: .3f}")

            if terminal:
                self.clear_buffer_()

        # return np.array(bboxes), np.array(scores)
        return np.array(bboxes), np.array(scores), np.array(all_scores)

    def predict_explore_(self, if_video=False):
        video_path = None
        bboxes, scores = [], []
        if if_video:
            assert self.env.index is not None
            index = self.env.index
            video_dir = os.path.join(self.params["video_save_dir"], f"test_{index}")
        for i in range(self.init_pos_grid.shape[0]):
            init_pos = self.init_pos_grid[i, :]  # (3,)
            if not self.env.is_in_vol_(init_pos, 2 * self.env.size):
                continue
            if if_video:
                video_path = create_param_dir(video_dir, f"agent_{i}.gif")
            bboxes_iter, scores_iter = [], []
            try:
                bboxes_iter, scores_iter = self.predict_explore_iter_(init_pos, if_video, video_path=video_path)
            except Exception:
                pass
            bboxes += bboxes_iter
            scores += scores_iter

        return np.array(bboxes), np.array(scores)

    def predict_explore_iter_(self, init_pos, if_video=False, **kwargs):
        num_steps = 0
        X_small, X_large, X_pos, X_size = self.env.reset()
        self.env.center = init_pos
        max_ep_len = self.agent.params["max_video_len"]
        (X_small, X_large, X_pos, X_size), _, _, _ = self.env.step((self.env.center, self.env.size))

        if if_video:
            img_clips = []
        bboxes, scores = [], []
        while True:
            num_steps += 1
            action = self.get_action((X_small, X_large, X_pos, X_size))
            (X_small, X_large, X_pos, X_size), _, done, _ = self.env.step(action)
            conf_score = self.classify_()
            if conf_score > self.params["conf_score_threshold_pred"]:
                start, end = center_size2start_end(X_pos, X_size)
                bboxes.append(np.concatenate([start, end], axis=0))
                scores.append(conf_score)
            # TODO: comment out
            self.env.render()
            if if_video:
                img_clips.append(self.env.render("rgb_array"))
            if num_steps == max_ep_len:
                done = True
            if done:
                self.clear_buffer_()
                break

        if if_video:
            assert kwargs.get("video_path", None) is not None
            record_gif(img_clips, kwargs["video_path"], fps=15)

        # list
        return bboxes, scores

    def classify_(self):
        # (T, 1, 1, P, P, P), (T, 1, 2P, 2P, 2P), (T, 1, 3)
        X_small, X_large, X_pos = self.obs_buffer["patches_small"], self.obs_buffer["patches_large"], \
                                  self.obs_buffer["rel_pos"]
        X_small = ptu.from_numpy(2 * X_small - 1)
        X_large = ptu.from_numpy(2 * X_large - 1)
        X_pos = ptu.from_numpy(X_pos)
        X_enc = self.agent.encoder(X_small, X_large, X_pos, X_pos[-1:, ...])  # (1, N_emb)
        X_clf = torch.softmax(self.agent.clf_head(X_enc), dim=-1)  # (1, 2)
        conf_score = X_clf[0, 1].item()

        return conf_score
