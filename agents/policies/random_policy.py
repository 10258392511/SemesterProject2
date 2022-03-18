import gym
import numpy as np

from SemesterProject2.agents.policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, env: gym.Env):
        super(RandomPolicy, self).__init__()
        self.env = env

    def get_action(self, obs) -> np.ndarray:
        return self.env.action_space.sample()
