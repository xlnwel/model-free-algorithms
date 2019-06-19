import numpy as np

from algo.off_policy.replay.utils import init_buffer, add_buffer, copy_buffer
from utility.debug_tools import assert_colorize

class LocalBuffer(dict):
    def __init__(self, args, state_space, action_dim):
        """ The following two fake data members are only used to complete the data pipeline """
        self.fake_ratio = np.zeros(args['local_capacity'])
        self.fake_ids = np.zeros(args['local_capacity'], dtype=np.int32)

        self.capacity = args['local_capacity']
        
        self.n_steps = args['n_steps']
        self.gamma = args['gamma']

        init_buffer(self, self.capacity, state_space, action_dim, True)
        
        self.idx = 0
        self.is_full = False

        if self.n_steps > 1:
            self.tb_capacity = args['tb_capacity']
            assert_colorize(self.tb_capacity >= self.n_steps, 'temporary buffer must be larger than n_steps')
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_space, action_dim, False)

    def __call__(self):
        while True:
            yield self.fake_ratio, self.fake_ids, (self['state'], self['action'], self['reward'], 
                                self['next_state'], self['done'], self['steps'])

    def reset(self):
        self.idx = 0
        self.is_full = False
        
    def add(self, state, action, reward, next_state, done):
        if self.n_steps > 1:
            add_buffer(self.tb, self.tb_idx, state, action, reward, 
                        done, self.n_steps, self.gamma)
            
            if not self.tb_full and self.tb_idx == self.n_steps - 1:
                self.tb_full = True
            self.tb_idx = (self.tb_idx + 1) % self.n_steps

            if done:
                self._merge(self.tb, self.tb_capacity if self.tb_full else self.tb_idx)
                self.tb_idx = 0
            elif self.tb_full:
                self._merge(self.tb, 1, self.tb_idx)
        else:
            add_buffer(self, self.idx, state, action, reward,
                        done, self.n_steps, self.gamma)
            self.idx += 1

    def _merge(self, local_buffer, length, start=0):
        assert_colorize(length < self.capacity, 'Temporary buffer is too large')
        end_idx = self.idx + length

        if end_idx > self.capacity:
            first_part = self.capacity - self.idx
            
            copy_buffer(self, self.idx, self.capacity, self.tb, start, start + first_part, dest_keys=False)
            # drop the excessive experiences
        else:
            copy_buffer(self, self.idx, end_idx, self.tb, start, start + length, dest_keys=False)
        
        if end_idx >= self.capacity:
            self.is_full = True

        self.idx = end_idx
