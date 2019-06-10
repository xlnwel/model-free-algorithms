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
        size = self.capacity if self.is_full else self.mem_idx
        indices = np.random.randint(0, size, self.batch_size)
        return [v[indices] for v in self.memory.values()]

    @override(Replay)
    def _merge(self, local_buffer, length, start=0):
        end_idx = self.mem_idx + length

        if end_idx > self.capacity:
            first_part = self.capacity - self.mem_idx
            second_part = length - first_part
            
            copy_buffer(self.memory, self.mem_idx, self.capacity, local_buffer, start, start + first_part)
            copy_buffer(self.memory, 0, second_part, local_buffer, start + first_part, start + length)
        else:
            copy_buffer(self.memory, self.mem_idx, end_idx, local_buffer, start, start + length)

        # memory is full, recycle buffer via FIFO
        if not self.is_full and end_idx >= self.capacity:
            print('Memory is fulll')
            self.is_full = True
        
        self.mem_idx = end_idx % self.capacity