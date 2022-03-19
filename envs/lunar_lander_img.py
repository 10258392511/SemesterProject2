import numpy as np
import gym
import SemesterProject2.scripts.configs as configs

from gym.envs.box2d.lunar_lander import LunarLander
from skimage.transform import resize


class LunarLanderImg(LunarLander):
    def __init__(self):
        super(LunarLanderImg, self).__init__()

    def preprocess_(self, img: np.ndarray):
        # img: (H, W, C)
        resized_shape = configs.lunar_lander_img_env_params["input_shape"]

        return resize(img, resized_shape)

    def step(self, action):
        _, reward, done, info = super().step(action)
        img_obs = super().render(mode="rgb_array")
        img_obs_out = self.preprocess_(img_obs)  # (H', W', C)

        return img_obs_out, reward, done, info
