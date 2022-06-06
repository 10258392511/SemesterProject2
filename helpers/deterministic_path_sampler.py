import numpy as np
import random

from itertools import cycle, product
from SemesterProject2.helpers.data_processing import VolumetricDataset
from SemesterProject2.helpers.utils import center_size2start_end, start_size2center_size, convert_to_rel_pos


class DeterministicPathSampler(object):
    def __init__(self, vol_ds: VolumetricDataset, params):
        """
        Works for both training and testing set.

        params: configs_ac.volumetric_env_params
        bash:
            grid_size
        """
        self.params = params
        self.vol_ds = vol_ds
        # to set in .sample()
        self.vol = None
        self.seg = None
        self.bboxes = None
        self.size = np.array(self.params["init_size"])  # xyz
        self.init_pos_grid = None  # xyz
        self.zeros_small = np.zeros(self.size)
        self.zeros_large = np.zeros(2 * self.size)
        self.step_size = (self.size * self.params["translation_scale"]).astype(int)
        self.xyz2ind = {"x": 2, "y": 1, "z": 0}
        self.sign2float = {"+": 1, "-": -1}

    def init_grid_(self):
        grid_w, grid_h, grid_d = self.params["grid_size"]
        D, H, W = self.vol.shape
        xx = np.linspace(0, W - 1, grid_w + 1)
        yy = np.linspace(0, H - 1, grid_h + 1)
        zz = np.linspace(0, D - 1, grid_d + 1)
        xx = np.clip((xx[:-1] + W / grid_w / 2).astype(int), 0, W - 1)
        yy = np.clip((yy[:-1] + H / grid_h / 2).astype(int), 0, H - 1)
        zz = np.clip((zz[:-1] + D / grid_d / 2).astype(int), 0, D - 1)
        self.init_pos_grid = list(product(xx, yy, zz))

    def sample(self):
        """
        Returns: list[(T, N, 1, P, P, P), (T, N, 1, 2P, 2P, 2P), (T, N, 3),
        (1, N, 1, P, P, P), (1, N, 1, 2P, 2P, 2P), (1, N, 3), (N,)]
        """
        ind = np.random.randint(len(self.vol_ds))
        self.vol, self.seg, self.bboxes = self.vol_ds[ind]
        self.init_grid_()
        path_len = np.random.randint(self.params["num_steps_to_memorize"] + 1)
        samples = []
        try:
            samples.append(self.sample_lesion_region_(path_len))
        except Exception as e:
            print(e)
        try:
            samples.append(self.sample_lesion_region_(path_len))
        except Exception as e:
            print(e)
        try:
            samples.append(self.sample_lesion_region_normal_(path_len))
        except Exception as e:
            print(e)
        try:
            samples.append(self.sample_normal_(path_len))
        except Exception as e:
            print(e)
        # TODO: consider make it a large batch
        return samples

    def sample_lesion_region_(self, path_len):
        """
        Returns: X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf
        (T, N, 1, P, P, P), (T, N, 1, 2P, 2P, 2P), (T, N, 3), (1, N, 1, P, P, P), (1, N, 1, 2P, 2P, 2P), (1, N, 3), (N,)
        """
        bbox = random.choice(self.bboxes)  # (6,)
        half_len = bbox.shape[0] // 2
        anchor_center, _ = start_size2center_size(bbox[:half_len], bbox[half_len:])  # xyz
        anchor_center = anchor_center[::-1]  # zyx
        assert self.seg[tuple(anchor_center)] == 1
        # X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf = [], [], [], [], [], [], []
        items = [[] for _ in range(7)]
        for mode in ("x+", "x-", "y+", "y-", "z+", "z-"):
            items_iter = self.sample_one_path_(anchor_center, path_len, mode, True)
            for i in range(len(items)):
                items[i].append(items_iter[i])

        for i in range(len(items)):
            if i == len(items) - 1:
                items[i] = np.array(items[i])  # list[bool] -> (N,)
            else:
                items[i] = np.concatenate(items[i], axis=1)  # e.g. list[(T, 1, 1, P, P, P)] -> (T, N, 1, P, P, P)

        return tuple(items)

    def sample_lesion_region_normal_(self, path_len):
        """
        Returns: X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf
        (T, N, 1, P, P, P), (T, N, 1, 2P, 2P, 2P), (T, N, 3), (1, N, 1, P, P, P), (1, N, 1, 2P, 2P, 2P), (1, N, 3), (N,)
        """
        bbox = random.choice(self.bboxes)  # (6,)
        half_len = bbox.shape[0] // 2
        anchor_center, _ = start_size2center_size(bbox[:half_len], bbox[half_len:])  # xyz
        anchor_center = anchor_center[::-1]  # zyx
        items = [[] for _ in range(7)]
        for mode in ("x+", "x-", "y+", "y-", "z+", "z-"):
            current_pos = anchor_center.copy()
            dim = self.xyz2ind[mode[0]]
            sign = self.sign2float[mode[1]]
            has_lesion = True
            while has_lesion:
                current_pos[dim] += -(self.step_size[dim] * sign).astype(int)
                has_lesion = (self.seg[tuple(current_pos)] == 1)
            items_iter = self.sample_one_path_(current_pos, path_len, mode, has_lesion)
            for i in range(len(items)):
                items[i].append(items_iter[i])

        for i in range(len(items)):
            if i == len(items) - 1:
                items[i] = np.array(items[i])  # list[bool] -> (N,)
            else:
                items[i] = np.concatenate(items[i], axis=1)  # e.g. list[(T, 1, 1, P, P, P)] -> (T, N, 1, P, P, P)

        return tuple(items)

    def sample_normal_(self, path_len):
        """
        Returns: X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf
        (T, N, 1, P, P, P), (T, N, 1, 2P, 2P, 2P), (T, N, 3), (1, N, 1, P, P, P), (1, N, 1, 2P, 2P, 2P), (1, N, 3), (N,)
        """
        anchor_center = np.array(random.choice(self.init_pos_grid))  # xyz
        has_lesion = True
        while has_lesion:
            center = anchor_center + np.random.randn(*anchor_center.shape) * self.params["init_perturb_std_ratio"] * self.size
            center = center.astype(int)  # xyz
            center = center[::-1]  # zyx
            has_lesion = (self.seg[tuple(center)] == 1.)
        print(f"anchor center for normal regions: {anchor_center}, has_lesion: {has_lesion}, seg: {self.seg[tuple(center)]}")

        # X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf = [], [], [], [], [], [], []
        items = [[] for _ in range(7)]
        for mode in ("x+", "x-", "y+", "y-", "z+", "z-"):
            # X_small_iter, X_large_iter, X_pos_iter, \
            # X_next_small_iter, X_next_large_iter, X_next_pos_iter, X_clf = self.sample_one_path_(center, path_len, mode, has_lesion)
            items_iter = self.sample_one_path_(center, path_len, mode, has_lesion)
            for i in range(len(items)):
                items[i].append(items_iter[i])

        for i in range(len(items)):
            if i == len(items) - 1:
                items[i] = np.array(items[i])  # list[bool] -> (N,)
            else:
                items[i] = np.concatenate(items[i], axis=1)  # e.g. list[(T, 1, 1, P, P, P)] -> (T, N, 1, P, P, P)

        return tuple(items)

    def sample_one_path_(self, end_pos, path_len, mode, has_lesion: bool):
        """
        end_pos: (3,), zyx, to be converted to xyz
        Returns: X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf
        (T, 1, 1, P, P, P), (T, 1, 1, 2P, 2P, 2P), (T, 1, 3), (1, 1, 1, P, P, P), (1, 1, 1, 2P, 2P, 2P), (1, 1, 3), bool
        """
        assert mode in ("x+", "x-", "y+", "y-", "z+", "z-")
        if not self.is_in_vol_(end_pos[::-1], 2 * self.size):
            return None, None, None, None, None, None
        X_clf = has_lesion
        X_small, X_large, X_pos = [], [], []
        X_next_pos = convert_to_rel_pos(end_pos[::-1], np.array(self.vol.shape[::-1])).reshape(1, 1, -1)  # (1, 1, 3)
        X_next_small = self.get_patch_by_center_size(end_pos[::-1], self.size).reshape(1, 1, 1, *self.size)  # (1, 1, 1, P, P, P)
        X_next_large = self.get_patch_by_center_size(end_pos[::-1], 2 * self.size).reshape(1, 1, 1, *(2 * self.size))  # (1, 1, 1, 2P, 2P, 2P)

        current_pos = end_pos.copy()
        dim = self.xyz2ind[mode[0]]
        sign = self.sign2float[mode[1]]

        for _ in range(path_len):
            # print(current_pos)
            current_pos[dim] += (self.step_size[::-1][dim] * sign).astype(int)
            if not self.is_in_vol_(current_pos[::-1], 2 * self.size):
                # print("not in vol")
                X_small_iter = self.zeros_small.copy()
                X_large_iter = self.zeros_large.copy()
            else:
                X_small_iter = self.get_patch_by_center_size(current_pos[::-1], self.size)  # (P, P, P)
                X_large_iter = self.get_patch_by_center_size(current_pos[::-1], 2 * self.size)  # (2P, 2P, 2P)
            X_small.append(X_small_iter.reshape(1, 1, 1, *self.size))  # list[(1, 1, 1, P, P, P)]
            X_large.append(X_large_iter.reshape(1, 1, 1, *(2 * self.size)))  # list[(1, 1, 1, 2P, 2P, 2P)]
            current_pos_rel = convert_to_rel_pos(current_pos[::-1], np.array(self.vol.shape[::-1]))  # (3,)
            X_pos.append(current_pos_rel.reshape(1, 1, -1))  # list[(1, 1, 3)]

        X_small.reverse()
        X_large.reverse()
        X_pos.reverse()
        X_small = np.concatenate(X_small, axis=0)  # (T, 1, 1, P, P, P)
        X_large = np.concatenate(X_large, axis=0)  # (T, 1, 1, 2P, 2P, 2P)
        X_pos = np.concatenate(X_pos, axis=0)  # (T, 1, 3)

        return X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, X_clf

    def get_patch_by_center_size(self, center, size):
        assert self.vol is not None
        # size = np.maximum(size, 1)
        start, end = center_size2start_end(center, size)

        return self.vol[start[2]:end[2], start[1]:end[1], start[0]:end[0]]

    def is_in_vol_(self, center, size):
        start, end = center_size2start_end(center, size)
        if np.all(start > 0) and np.all(end <= np.array(self.vol.shape[::-1])):
            return True
        return False
