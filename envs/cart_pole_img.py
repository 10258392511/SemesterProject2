import numpy as np

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from gym.envs.classic_control import CartPoleEnv


resized_shape = (50, 75)

class CartPoleImg(CartPoleEnv):
    def __init__(self):
        super(CartPoleImg, self).__init__()
        self.obs_seq = None
        self.reset()

    def preprocess_(self, img):
        img_obs = rgb2gray(img)
        # img_obs = resize(img_obs, resized_shape).transpose(2, 0, 1)
        img_obs = resize(img_obs, resized_shape)

        return img_obs[None, ...]

    def step(self, action):
        obs, reward, terminal, info = super().step(action)
        img_obs = super().render(mode="rgb_array")
        img_obs = self.preprocess_(img_obs)
        self.obs_seq.append(img_obs)
        if len(self.obs_seq) > 2:
            self.obs_seq.popleft()
        img_obs_out = self.obs_seq[1] - self.obs_seq[0]

        return img_obs_out, reward, terminal, info

    def reset(self):
        super().reset()
        self.obs_seq = deque()
        img_obs = super().render(mode="rgb_array")
        img_obs = self.preprocess_(img_obs)
        self.obs_seq.append(img_obs)

        return img_obs
