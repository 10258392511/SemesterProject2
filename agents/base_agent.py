class BaseAgent(object):
    def train(self) -> dict:
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, filename, *args, **kwargs):
        raise NotImplementedError

    def load(self, filename, *args, **kwargs):
        raise NotImplementedError
