import numpy as np

class BasePolicy(object):
    def get_action(self, obs) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs, acts, **kwargs) -> dict:
        raise NotImplementedError

    def save(self, filepath) -> None:
        raise NotImplementedError
