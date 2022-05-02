from SemesterProject2.agents.policies.base_policy import BasePolicy
from SemesterProject2.agents.vit_agent import ViTAgent


class ViTPolicy(BasePolicy):
    def __init__(self, agent: ViTAgent):
        super(ViTPolicy, self).__init__()
        self.agent = agent
        self.terminal = False
        self.obs_buffer = []

    def get_action(self, obs) -> np.ndarray:
        pass

    def convert_buffer_to_obs_(self):
        pass
