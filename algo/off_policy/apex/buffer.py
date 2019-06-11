import numpy as np

from algo.off_policy.replay.utils import init_buffer, add_buffer, copy_buffer
from utility.utils import assert_colorize

class LocalBuffer(dict):
    def __init__(self, args, state_space, action_dim):
        """ The following two fake data members are only used to complete the data pipeline """
        self.fake_ratio = np.zeros(args['local_capacity'])
        self.fake_ids = np.zeros(args['local_capacity'], dtype=np.int32)

        self.capacity = args['local_capacity']
        
        self.n_steps = args['n_steps']
        self.gamma = args['gamma']

        init_buffer(self, self.capacity, state_space, action_dim, True)
        self.reset()

        if self.n_steps > 1:
            self.tb_capacity = args['tb_capacity']
            assert_colorize(self.tb_capacity >= self.n_steps, 'temporary buffer must be larger than n_steps')
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_space, action_dim, False)

    @property
    def is_full(self):
        return self.idx == self.capacity

    def __call__(self):
        while True:
            yield self.fake_ratio, self.fake_ids, (self['state'], self['action'], self['reward'], 
                                self['next_state'], self['done'], self['steps'])

    def reset(self):
        self.idx = 0
        
    def add(self, state, action, reward, next_state, done):
        if self.n_steps > 1:
            add_buffer(self.tb, self.tb_idx, state, action, reward, 
                        next_state, done, self.n_steps, self.gamma)
            
            if not self.tb_full and self.tb_idx == self.n_steps - 1:
                self.tb_full = True
            self.tb_idx = (self.tb_idx + 1) % self.n_steps

            if done:
                self._merge(self.tb_capacity if self.tb_full else self.tb_idx)
            elif self.tb_full:
                copy_buffer(self, self.idx, self.idx+1, self.tb, self.tb_idx, self.tb_idx+1)
                self.idx += 1
        else:
            add_buffer(self, self.idx, state, action, reward,
                        next_state, done, self.n_steps, self.gamma)
            self.idx += 1

    def _merge(self, length, start=0):
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