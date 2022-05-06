import numpy as np
import cv2 as cv

from typing import Optional, Tuple, Union
from gym.core import Env
from SemesterProject2.helpers.data_processing import VolumetricDataset
from SemesterProject2.helpers.utils import center_size2start_end, start_size2start_end, start_size2center_size


class Volumetric(Env):
    """
    observation: (patch_small, patch_large, center, size)
    action: (center, size)
    reward: fuel_cost (< 0) + dice_reward
    """
    def __init__(self, params):
        """
        params:
            bash: hdf5_filename, data_splitted_filename, mode, seed
            env: max_ep_len, init_size, dice_reward_weighting, fuel_cost
        coord: x, y, z
        """
        assert params["mode"] in ("train", "test")
        super(Volumetric, self).__init__()
        self.params = params
        self.dataset = VolumetricDataset(self.params["hdf5_filename"], self.params["data_splitted_filename"],
                                         mode=self.params["mode"])
        self.center = None
        self.size = np.array(self.params["init_size"])
        self.vol = None
        self.seg = None
        self.bbox_coord = None
        self.closest_lesion_seg = None
        self.time_step = 0
        self.dice_score_small, self.dice_score_large = 0, 0
        self.cv_window_name = "VolumetricEnv"
        self.sample_index = None

    def close(self):
        # close .cv_window
        # assert self.cv_window is not None
        try:
            cv.destroyWindow(self.cv_window_name)
        except Exception:
            pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union:
        seed = self.params["seed"] if seed is None else seed
        # super().reset(seed=seed)

        # sample one volume from .dataset
        np.random.seed(seed)
        index = np.random.randint(len(self.dataset))
        self.vol, self.seg, self.bbox_coord = self.dataset[index]
        self.center = None
        self.size = np.array(self.params["init_size"])
        self.closest_lesion_seg = None
        self.time_step = 0
        self.dice_score_small, self.dice_score_large = 0, 0
        self.sample_index = index

        while True:
            vol_center = self.get_vol_center_()
            # sample .center from N(.center, .size ** 2)
            self.center = (vol_center + np.random.randn(*self.size.shape) * self.size).astype(np.int)
            patch_small = self.get_patch_by_center_size(self.center, self.size)
            patch_large = self.get_patch_by_center_size(self.center, self.size * 2)

            if self.is_in_vol_(self.center, 2 * self.size):
                break

        return patch_small, patch_large, self.center, self.size

    def render(self, mode="human"):
        """
        Renders the slice the agent is at. Agent glimpses are in blue, and if there is lesion within the slice, it's
        shown in red.
        """
        assert mode in ["human", "rgb_array"]
        z = self.center[-1]
        img_slice = (self.vol[z, ...] * 255).astype(np.uint8)
        img_slice = cv.cvtColor(img_slice, cv.COLOR_GRAY2RGB)
        start_small, end_small = center_size2start_end(self.center, self.size)
        start_large, end_large = center_size2start_end(self.center, 2 * self.size)
        glimpses_slice = [(start_small[:2], end_small[:2]), (start_large[:2], end_large[:2])]
        for glimpse_slice in glimpses_slice:
            # blue
            cv.rectangle(img_slice, glimpse_slice[0], glimpse_slice[1], color=(0, 0, 255),
                         thickness=2, lineType=cv.LINE_AA)
        seg_slice = self.seg[z, ...]
        seg_slice_mask = (seg_slice > 0)

        # red
        img_slice[seg_slice_mask, 0] = 255
        img_slice[seg_slice_mask, 1:] = 0

        bbox_coord_render = []
        for bbox_coord in self.bbox_coord:
            # (x_min, y_min, z_min, x_size, y_size, z_size)
            bbox_coord_half_len = len(bbox_coord) // 2
            bbox_start, bbox_end = start_size2start_end(bbox_coord[:bbox_coord_half_len],
                                                        bbox_coord[bbox_coord_half_len:])
            if bbox_start[-1] <= z <= bbox_end[-1]:
                cv.rectangle(img_slice, bbox_start[:2], bbox_end[:2], color=(255, 0, 0), thickness=2,
                             lineType=cv.LINE_AA)

        # purple: closest legion mask
        if self.closest_lesion_seg is not None:
            mask = (self.closest_lesion_seg[z, ...] > 0)
            img_slice[mask, 0] = 255
            img_slice[mask, 2] = 255
            img_slice[mask, 1] = 0

        if mode == "rgb_array":
            return img_slice

        else:
            cv.namedWindow(self.cv_window_name)
            cv.imshow(self.cv_window_name, img_slice[..., ::-1])
            cv.waitKey(10)

    def step(self, action) -> Tuple:
        """
        action: (center, size), xyz-coord
        """
        self.time_step += 1
        act_center, act_size = action
        done = False
        reward = self.params["fuel_cost"]
        if not self.is_in_vol_(act_center, act_size * 2):
            done  = True
        if self.time_step > self.params["max_ep_len"]:
            done = True
        if not done:
            self.center, self.size = act_center, act_size
        patch_small, patch_large = self.get_patch_by_center_size(self.center, self.size), \
                                   self.get_patch_by_center_size(self.center, self.size * 2)
        assert np.all(np.array(patch_small.shape) > 0) and np.all(np.array(patch_large.shape) > 0), "shape <= 0"

        if done:
            reward += (self.dice_score_small + self.dice_score_large) * self.params["dice_reward_weighting"]
            return (patch_small, patch_large, self.center, self.size), reward, done, \
                   {"dice_score_small": self.dice_score_small, "dice_score_large": self.dice_score_large}

        self.compute_closest_legion_seg_()  # set .closest_legion_seg
        dice_score_small = self.compute_dice_score_(self.center, self.size)
        dice_score_large = self.compute_dice_score_(self.center, self.size * 2)

        dice_score_small_sign = 1 if dice_score_small > self.dice_score_small else -1
        self.dice_score_small = dice_score_small
        dice_score_large_sign = 1 if dice_score_large > self.dice_score_large else -1
        self.dice_score_large = dice_score_large
        reward += dice_score_small_sign + dice_score_large_sign

        return (patch_small, patch_large, self.center, self.size), reward, done, {"dice_score_small": dice_score_small,
                                                                                  "dice_score_large": dice_score_large}

    def is_in_vol_(self, center, size):
        start, end = center_size2start_end(center, size)
        if np.all(start > 0) and np.all(end < np.array(self.vol.shape[::-1])):
            return True
        return False

    def compute_dice_score_(self, patch_center, patch_size):
        patch_mask = self.convert_bbox_coord_to_mask_(np.concatenate([patch_center, patch_size], axis=-1))
        assert self.closest_lesion_seg is not None
        seg_selected = self.closest_lesion_seg
        intersection = (patch_mask * seg_selected).sum()  # (D, H, W) * (D, H, W) -> sum(.) -> (1,)
        dice_score = 2 * (intersection) / (patch_mask.sum() + seg_selected.sum())

        return dice_score

    def compute_closest_legion_seg_(self):
        """
        Returns selected seg.
        """
        closest_dist, closest_bbox = float("inf"), None
        for bbox_coord in self.bbox_coord:
            bbox_coord_half_len = len(bbox_coord) // 2
            start, size = bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:]
            center, _ = start_size2center_size(start, size)
            dist = np.linalg.norm(self.center - center)
            if dist < closest_dist:
                closest_dist = dist
                closest_bbox = bbox_coord

        mask = self.convert_bbox_coord_to_mask_(closest_bbox)
        seg = self.seg * mask
        self.closest_lesion_seg = seg

    def convert_bbox_coord_to_mask_(self, bbox_coord):
        """
        bbox_coord: (x_min, y_min, z_min, x_size, y_size, z_size), np.ndarray
        """
        bbox_coord_half_len = len(bbox_coord) // 2
        start, size = bbox_coord[:bbox_coord_half_len], bbox_coord[bbox_coord_half_len:]
        end = (start + size).astype(np.int)
        mask = np.zeros_like(self.seg)
        mask[start[2]:end[2], start[1]:end[1], start[0]:end[0]] = 1

        return mask

    def get_vol_center_(self):
        assert self.vol is not None

        return (np.array(self.vol.shape) / 2).astype(np.int)

    def get_patch_by_center_size(self, center, size):
        assert self.vol is not None
        start, end = center_size2start_end(center, size)

        return self.vol[start[2]:end[2], start[1]:end[1], start[0]:end[0]]
