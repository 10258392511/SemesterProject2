import torch
import SemesterProject2.helpers.pytorch_utils as ptu
import SemesterProject2.scripts.configs_network as configs_network

from SemesterProject2.agents.base_agent import BaseAgent
from SemesterProject2.helpers.modules.vit_agent_modules import Encoder, MLPHead
from SemesterProject2.helpers.replay_buffer_vit import ReplayBuffer


class ViTAgent(BaseAgent):
    def __init__(self, params):
        """
        params:
        configs.vit_agent_params:
            num_target_steps, num_grad_steps_per_target_update, lam_cls, replay_buffer_size, dice_score_small_th
        As keys:
            encoder_params, *_head_params
        """
        super(ViTAgent, self).__init__()
        self.params = params
        self.encoder = Encoder(self.params["encoder_params"]).to(ptu.device)
        self.patch_pred_head = MLPHead(self.params["patch_pred_head_params"]).to(ptu.device)
        self.critic_head = MLPHead(self.params["critic_head_params"]).to(ptu.device)
        self.actor_head = MLPHead(self.params["actor_head_params"]).to(ptu.device)
        self.replay_buffer = ReplayBuffer(self.params)

    def train(self, paths) -> dict:
        pass

    def pre_train(self, paths) -> dict:
        pass

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        """
        TODO
        Sample rollouts: list[Path]; delegated to .replay_buffer
        """
        pass

    def update_critic(self, paths) -> dict:
        pass

    def compute_v_vals(self, obs) -> dict:
        pass

    def update_actor(self, paths) -> dict:
        pass

    def compute_likelihood_and_penalty_(self, paths):
        """
        Returns:
            bbox_log_lh: (T,), cls_log_lh: (T,), cls_penalty: (1,)
        """
        pass

    def compute_rewards_(self, paths):
        pass

