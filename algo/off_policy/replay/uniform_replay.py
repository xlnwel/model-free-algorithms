import threading
import numpy as np

from utility.decorators import override
from algo.off_policy.replay.basic_replay import Replay
from algo.off_policy.replay.utils import add_buffer, copy_buffer


class UniformReplay(Replay):
    """ Interface """
    def __init__(self, args, state_space, action_space):
        super(args, state_space, action_space)

    @override(Replay)
    def add(self, state, action, reward, next_state, done):
        super()._add(state, action, reward, next_state, done)

    """ Implementation """
    @override(Replay)
    def _sample(self):
        indices = np.random.randint(0, self.capacity, self.batch_size)
        return [v[indices] for v in self.memory.values()]

    @override(Replay)
    def _merge(self, local_buffer, length, start=0):
        end_idx = self.exp_id + length

        if end_idx > self.capacity:
            first_part = self.capacity - self.exp_id
            second_part = length - first_part
            
            copy_buffer(self.memory, self.exp_id, self.capacity, local_buffer, start, start + first_part)
            copy_buffer(self.memory, 0, second_part, local_buffer, start + first_part, start + length)
        else:
            copy_buffer(self.memory, self.exp_id, end_idx, local_buffer, start, start + length)

        # memory is full, recycle buffer via FIFO
        if not self.is_full and end_idx >= self.capacity:
            print('Memory is fulll')
            self.is_full = True
        
        self.exp_id = end_idx % self.capacity