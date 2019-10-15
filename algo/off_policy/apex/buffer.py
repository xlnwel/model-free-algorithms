import numpy as np

from algo.off_policy.replay.utils import init_buffer, add_buffer, copy_buffer
from utility.debug_tools import assert_colorize


class LocalBuffer(dict):
    """ Local buffer only stores one episode of transitions """
    def __init__(self, args, state_space, action_dim):
        self.capacity = args['local_capacity']
        self.n_steps = args['n_steps']
        self.gamma = args['gamma']

        # The following two fake data members are only used to complete the data pipeline
        self.fake_ratio = np.zeros(self.capacity)
        self.fake_ids = np.zeros(self.capacity, dtype=np.int32)

        init_buffer(self, self.capacity, state_space, action_dim, True, extra_state=1)

        self.reward_scale = args['reward_scale'] if 'reward_scale' in args else 1
        
        self.idx = 0

    def __call__(self):
        """ call to compute priorities """
        while True:
            yield self.fake_ratio, self.fake_ids, (self['state'][:-1], self['action'], self['reward'],
                                                    self['state'][1:], self['done'], self['steps'])

    def reset(self):
        self.idx = 0
        
    def add_data(self, state, action, reward, done):
        """ Add experience to local buffer, return True if local buffer is full, otherwise false """
        add_buffer(self, self.idx, state, action, reward, 
                    done, self.n_steps, self.gamma)
        
        self.idx = self.idx + 1

    def add_last_state(self, state):
        self['state'][self.idx] = state