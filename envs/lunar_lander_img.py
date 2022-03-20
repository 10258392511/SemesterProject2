import numpy as np
import gym
# import SemesterProject2.scripts.configs as configs

from gym.envs.box2d.lunar_lander import LunarLander
from skimage.transform import resize

resized_shape = (100, 150)

class LunarLanderImg(LunarLander):
    def __init__(self):
        super(LunarLanderImg, self).__init__()

    def preprocess_(self, img: np.ndarray):
        # img: (H, W, C)
        # resized_shape = configs.lunar_lander_img_env_params["input_shape"]
        img_out = resize(img, resized_shape)

        return np.transpose(img_out, (2, 0, 1))

    def step(self, action):
        _, reward, done, info = super().step(action)
        img_obs = super().render(mode="rgb_array")
        # print(img_obs.shape)
        img_obs_out = self.preprocess_(img_obs)  # (H', W', C)
        # self.close()

        return img_obs_out, reward, done, info
