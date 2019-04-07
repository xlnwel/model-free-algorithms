from abc import ABCMeta, abstractclassmethod
import numpy as np
from replay.utils import add_buffer, copy_buffer


class Replay(metaclass=ABCMeta):
    def __init__(self, args, state_space, action_dim):
        self.memory = {}

        # params for general replay buffer
        self.capacity = int(float(args['capacity']))
        self.min_size = int(float(args['min_size']))
        self.batch_size = args['batch_size']

        self.n_steps = args['n_steps']
        self.gamma = args['gamma']

        self.is_full = False
        self.exp_id = 0

        def init_buffer(buffer, capacity, state_space, action_dim, has_priority):
            target_buffer = {'priority': np.zeros((capacity, 1))} if has_priority else {}
            target_buffer.update({
                'state': np.zeros((capacity, *state_space)),
                'action': np.zeros((capacity, action_dim)),
                'reward': np.zeros((capacity, 1)),
                'next_state': np.zeros((capacity, *state_space)),
                'done': np.zeros((capacity, 1)),
                'steps': np.zeros((capacity, 1))
            })

            buffer.update(target_buffer)

        init_buffer(self.memory, self.capacity, state_space, action_dim, False)

        # Code for single agent
        if self.n_steps > 1:
            self.temporary_buffer = {}
            init_buffer(self.temporary_buffer, self.n_steps, state_space, action_dim, True)
            self.tb_idx = 0
            self.tb_full = False

    @property
    def good_to_learn(self):
        return len(self) >= self.min_size

    def __len__(self):
        return self.capacity if self.is_full else self.exp_id

    def __call__(self):
        while True:
            yield self.sample()
            
    @abstractclassmethod
    def sample(self):
        raise NotImplementedError

    @abstractclassmethod
    def merge(self, local_buffer, length, start=0):
        raise NotImplementedError

    # Code for single agent
    def add(self, state, action, reward, next_state, done):
        if self.n_steps > 1:
            add_buffer(self.temporary_buffer, self.tb_idx, state, action, reward, 
                        next_state, done, self.n_steps, self.gamma)
            
            if not self.tb_full and self.tb_idx == self.n_steps - 1:
                self.tb_full = True
            self.tb_idx = (self.tb_idx + 1) % self.n_steps

            if done:
                # flush all elements in temporary buffer to memory if an episode is done
                self.merge(self.temporary_buffer, self.n_steps if self.tb_full else self.tb_idx)
                self.tb_full = False
                self.tb_idx = 0
            elif self.tb_full:
                # add the ready experience in temporary buffer to memory
                self.merge(self.temporary_buffer, 1, self.tb_idx)
        else:
            add_buffer(self.memory, self.exp_id, state, action, reward,
                        next_state, done, self.n_steps, self.gamma)
            self.exp_id += 1