import numpy as np
import cv2 as cv

from gym.core import Env
from SemesterProject2.helpers.data_processing import VolumetricDataset
from SemesterProject2.helpers.utils import center_size2start_end


class Volumetric(Env):
    def __init__(self, params):
        """
        params:
            bash: hdf5_filename, data_splitted_filename, mode, seed
            env: max_ep_len, init_size, dice_reward_weighting
        """
        assert params["mode"] in ("train", "test")
        super(Volumetric, self).__init__()
        self.params = params
        self.dataset = VolumetricDataset(self.params["hdf5_filename"], self.params["data_splitted_filename"],
                                         mode=self.params["mode"])
        self.center = None
        self.size = None
        self.vol = None
        self.seg = None
        if self.params["mode"] == "test":
            self.bbox = None
            self.bbox_coord = None
        self.time_step = 0
        self.dice_score_small, self.dice_score_large = 0, 0
        self.cv_window = None

    def close(self):
        # TODO: close .cv_window
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        seed = self.params["seed"] if seed is None else seed
        super().reset(seed)
        pass

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"]
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        action: (center, size)
        """
        pass

    def get_patch_by_bbox_(self, center, size):
        pass

    def compute_dice_score(self, patch):
        # TODO: use monai's metric
        pass
