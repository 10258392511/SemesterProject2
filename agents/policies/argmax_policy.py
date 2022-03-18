import numpy as np
import torch

from SemesterProject2.agents.policies.base_policy import BasePolicy
from SemesterProject2.agents.base_agent import BaseAgent


class ArgmaxPolicy(BasePolicy):
    def __init__(self, agent: BaseAgent):
        super(BasePolicy, self).__init__()
        self.agent = agent

    @torch.no_grad()
    def get_action(self, obs) -> np.ndarray:
        return self.agent.get_action(obs, eps=0)
