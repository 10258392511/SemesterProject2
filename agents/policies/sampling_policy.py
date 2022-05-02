import numpy as np
import SemesterProject2.scripts.configs_ac as configs_ac

from SemesterProject2.agents.policies.base_policy import BasePolicy


class SamplingPolicy(BasePolicy):
    def __init__(self, params):
        """
        params: (from configs_ac.volumetric_sampling_policy_args) max_ep_len, translation_scale, size_scale
        """
        super(SamplingPolicy, self).__init__()
        self.params = params
        self.size_last_signs = [0, 0, 0]
        self.terminal = False

    def get_action(self, obs) -> np.ndarray:
        """
        obs: (patch_small, patch_large, center, size)
        """
        _, _, center, size = obs
        next_center, next_size = center.astype(np.float), size.astype(np.float)
        for dim in range(next_center.shape[0]):
            sample = np.random.rand()
            if sample < 1 / 3:
                sign = -1
            elif sample > 2 / 3:
                sign = 1
            else:
                sign = 0
            next_center[dim] = next_center[dim] + sign * self.params["translation_scale"] * next_size[dim]

        for dim in range(next_size.shape[0]):
            sample = np.random.rand()
            if self.size_last_signs[dim] == -1:
                sign = 1
            elif self.size_last_signs[dim] == 1:
                sign = 0
            else:
                if sample < 1 / 3:
                    sign = -1
                elif sample > 2 / 3:
                    sign = 1
                else:
                    sign = 0
            next_size[dim] *= (1 + sign * self.params["size_scale"])
            self.size_last_signs[dim] = sign

        return next_center.astype(np.int), next_size.astype(np.int)
