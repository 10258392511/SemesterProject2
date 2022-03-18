import numpy as np

from SemesterProject2.helpers.utils import convert_list_of_rollouts


class ReplayBuffer(object):
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.paths = []
        self.obs, self.acts, self.concatenated_rews, self.unconcatenated_rews, self.next_obs, self.terminals = \
        None, None, None, None, None, None

    def __repr__(self):
        if self.obs is None:
            out_str = "ReplayBuffer(uninitialized)"
        else:
            out_str = f"ReplayBuffer(obs: {self.obs.shape}, acts: {self.acts.shape}, " \
                      f"concat_rews: {self.concatenated_rews.shape}, unconcat_rews: {len(self.unconcatenated_rews)}, " \
                      f"next_obs: {self.next_obs.shape}, terminals: {self.terminals.shape})"

        return out_str

    def add_rollouts(self, paths):
        for path in paths:
            self.paths.append(path)

        obs, actions, concatenated_rewards, next_obs, terminals, unconcatenated_rewards = \
            convert_list_of_rollouts(paths)  # (P, N_obs) &etc.
        if self.obs is None:
            self.obs = obs[-self.max_size:, ...]
            self.acts = actions[-self.max_size:, ...]
            self.concatenated_rews = concatenated_rewards[-self.max_size:, ...]
            self.next_obs = next_obs[-self.max_size:, ...]
            self.terminals = terminals[-self.max_size:, ...]
            self.unconcatenated_rews = unconcatenated_rewards[-self.max_size:]  # list, TODO: how to use
        else:
            self.obs = np.concatenate([self.obs, obs], axis=0)[-self.max_size:, ...]
            self.acts = np.concatenate([self.acts, actions], axis=0)[-self.max_size:, ...]
            self.concatenated_rews = np.concatenate([self.concatenated_rews, concatenated_rewards], axis=0)[-self.max_size:, ...]
            self.next_obs = np.concatenate([self.next_obs, next_obs], axis=0)[-self.max_size:, ...]
            self.terminals = np.concatenate([self.terminals, terminals], axis=0)[-self.max_size:, ...]
            self.unconcatenated_rews += unconcatenated_rewards
            self.unconcatenated_rews = self.unconcatenated_rews[-self.max_size:]

    def can_sample(self, batch_size):
        if self.obs is None or self.obs.shape[0] < batch_size:
            return False

        return True

    def sample_random_data(self, batch_size):
        # Only consider "concatenated_rewards"
        indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        # print(indices)

        return self.obs[indices, ...], self.acts[indices, ...], self.concatenated_rews[indices, ...], \
               self.next_obs[indices, ...], self.terminals[indices, ...]

    def sample_recent_data(self, batch_size):
        return self.obs[-batch_size:, ...], self.acts[-batch_size:, ...], self.concatenated_rews[-batch_size:, ...], \
               self.next_obs[-batch_size:, ...], self.terminals[-batch_size:, ...]
