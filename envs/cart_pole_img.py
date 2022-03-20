import numpy as np

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from gym.envs.classic_control import CartPoleEnv


resized_shape = (48, 48)

class CartPoleImg(CartPoleEnv):
    def __init__(self):
        super(CartPoleImg, self).__init__()
        self.obs_seq = None
        self.reset()

    def preprocess_(self, img):
        cart_roi = self.get_cart_roi_(img)

        img_obs = rgb2gray(cart_roi)
        img_obs = resize(img_obs, resized_shape)
        return img_obs[None, ...]

        # img_obs = cart_roi
        # img_obs = resize(img_obs, resized_shape).transpose(2, 0, 1)
        # return img_obs



    def get_cart_roi_(self, img_obs):
        # [-2.4, 2.4] -> [0, 4.8] -> [0, screen_width]
        screen_width = 600
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        cart_loc = int((self.state[0] + self.x_threshold) * scale)
        sky_line = 170
        ground_line = 320
        cart_half_width = (ground_line - sky_line) // 2
        cart_roi = np.zeros((ground_line - sky_line + 1, 2 * cart_half_width + 1, 3))
        left_most = cart_loc - cart_half_width
        right_most = cart_loc + cart_half_width + 1
        if left_most < 0 and right_most >= 0:
            cart_roi_patch = img_obs[sky_line:ground_line + 1, 0:right_most]
            cart_roi[:, -cart_roi_patch.shape[1]:, :] = cart_roi_patch
        elif right_most > screen_width and left_most <= screen_width:
            cart_roi_patch = img_obs[sky_line:ground_line + 1, left_most:screen_width]
            cart_roi[:, 0:cart_roi_patch.shape[1], :] = cart_roi_patch
        elif left_most >= 0 and right_most <= screen_width:
            cart_roi = img_obs[sky_line:ground_line + 1, left_most:right_most, :]

        return cart_roi

    def step(self, action):
        obs, reward, terminal, info = super().step(action)
        img_obs = super().render(mode="rgb_array")
        img_obs = self.preprocess_(img_obs)
        self.obs_seq.append(img_obs)
        if len(self.obs_seq) > 2:
            self.obs_seq.popleft()
        img_obs_out = (self.obs_seq[1] - self.obs_seq[0])  # (C, H, W)
        # (C, 1, 1)
        img_obs_min, img_obs_max = np.min(img_obs_out, axis=(1, 2), keepdims=True), \
                                   np.max(img_obs_out, axis=(1, 2), keepdims=True)
        img_obs_out = (img_obs_out - img_obs_min) / (img_obs_max - img_obs_min + 1e-5)
        img_obs_out = 2 * img_obs_out - 1
        # img_obs_out = img_obs
        # img_obs_out = np.concatenate(self.obs_seq, axis=0)

        return img_obs_out, reward, terminal, info

    def reset(self):
        super().reset()
        self.obs_seq = deque()
        img_obs = super().render(mode="rgb_array")
        img_obs = self.preprocess_(img_obs)
        self.obs_seq.append(img_obs)
        # img_obs_out = np.concatenate([np.zeros_like(img_obs), img_obs], axis=0)
        img_obs_out = img_obs

        return img_obs_out
