import random
from collections import deque
import numpy as np
from replay.replaybuffer import ReplayBuffer

class UniformReplay(ReplayBuffer):
    # Interface
    def __init__(self, args, sample_size, n_steps=1, gamma=1):
        super().__init__(args, sample_size, n_steps=n_steps, gamma=gamma)
        self.memory = deque(maxlen=self.capacity)
        
    def sample(self):
        assert self.good_to_learn, 'There are not sufficient transitions in buffer to learn'
        exps = random.sample(self.memory, self.sample_size)
        
        # None here for being consistent with the prioritized API
        return None, self._unpack_samples(exps)

    def __len__(self):
        return len(self.memory)
 
    # Implementation
    def _add_exp(self, exp):
        self.memory.append(exp)