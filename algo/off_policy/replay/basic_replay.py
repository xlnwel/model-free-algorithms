import threading
import numpy as np

from algo.off_policy.replay.utils import add_buffer, copy_buffer
from algo.off_policy.replay.utils import init_buffer
from utility.utils import assert_colorize


class Replay:
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

        init_buffer(self.memory, self.capacity, state_space, action_dim, False)

        # Code for single agent
        if self.n_steps > 1:
            self.temporary_buffer = {}
            init_buffer(self.temporary_buffer, self.n_steps, state_space, action_dim, True)
            self.tb_idx = 0
            self.tb_full = False
        
        # locker used to avoid conflict introduced by tf.data.Dataset and multi-agent
        self.locker = threading.Lock()

    @property
    def good_to_learn(self):
        return len(self) >= self.min_size

    def __len__(self):
        return self.capacity if self.is_full else self.exp_id

    def __call__(self):
        while True:
            yield self.sample()
            
    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:
            samples = self._sample()

        return samples

    def merge(self, local_buffer, length, start=0):
        assert_colorize(length < self.capacity, 'Local buffer is too large')
        with self.locker:
            self._merge(local_buffer, length, start)

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

    def _sample(self):
        raise NotImplementedError

    def _merge(self, local_buffer, length, start=0):
        raise NotImplementedError
