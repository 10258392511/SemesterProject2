import numpy as np
import random
import SemesterProject2.scripts.configs_ac as configs_ac

from itertools import product
from SemesterProject2.agents.policies.base_policy import BasePolicy


class FixedSizeSampingPolicy(BasePolicy):
    def __init__(self, params):
        """
        params: (from configs_ac.volumetric_sampling_policy_args) max_ep_len, translation_scale, size_scale
        """
        super(FixedSizeSampingPolicy, self).__init__()
        self.params = params
        self.terminal = False  # Not used
        self.action_space = list(product([-self.params["translation_scale"], 0, self.params["translation_scale"]],
                                         repeat=3))
        self.action_space = [np.array(action) for action in self.action_space if action != (0, 0, 0)]

    def get_action(self, obs):
        """
        All: ndarray
        obs: X_small: (P, P, P), X_large: (2P, 2P, 2P), center: (3,), size: (3,)
        """
        _, _, center, size = obs
        action = random.choice(self.action_space)
        next_center = (center + action * size).astype(int)
        next_size = size

        return next_center, next_size

    def clear_buffer(self):
        pass
