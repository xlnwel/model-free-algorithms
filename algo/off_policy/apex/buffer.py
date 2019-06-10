import numpy as np

from algo.off_policy.replay.utils import init_buffer, add_buffer, copy_buffer


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
            self.tb = {}
            init_buffer(self.tb, self.n_steps, state_space, action_dim, False)
            self.tb_idx = 0
            self.tb_full = False

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

            if self.tb_full:
                copy_buffer(self, self.idx, self.idx+1, self.tb, self.tb_idx, self.tb_idx+1)
                self.idx += 1
        else:
            add_buffer(self, self.idx, state, action, reward,
                        next_state, done, self.n_steps, self.gamma)
            self.idx += 1
