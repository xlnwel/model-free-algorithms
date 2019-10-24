import numpy as np

from utility.decorators import override
from algo.off_policy.replay.basic_replay import Replay
from algo.off_policy.replay.utils import add_buffer, copy_buffer


class UniformReplay(Replay):
    """ Interface """
    def __init__(self, args, obs_space):
        super().__init__(args, obs_space)

    @override(Replay)
    def add(self, obs, action, reward, done):
        super()._add(obs, action, reward, done)

    """ Implementation """
    @override(Replay)
    def _sample(self):
        size = self.capacity if self.is_full else self.mem_idx
        indexes = np.random.randint(0, size, self.batch_size)
        
        samples = self._get_samples(indexes)

        return samples
