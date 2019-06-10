import threading
import numpy as np

from utility.utils import assert_colorize
from algo.off_policy.replay.utils import init_buffer, add_buffer, copy_buffer


class Replay:
    """ Interface """
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
            self.tb_capacity = self.n_steps
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_space, action_dim, True)
        
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

    def add(self):
        # locker should be handled in implementation
        raise NotImplementedError

    """ Implementation """
    def _add(self, state, action, reward, next_state, done):
        """ add is only used for single agent, no multiple adds are expected to run at the same time
            but it may fight for resource with self.sample if background learning is enabled """
        if self.n_steps > 1:
            add_buffer(self.tb, self.tb_idx, state, action, reward, 
                        next_state, done, self.n_steps, self.gamma)
            
            if not self.tb_full and self.tb_idx == self.tb_capacity - 1:
                self.tb_full = True
            self.tb_idx = (self.tb_idx + 1) % self.tb_capacity

            if done:
                # flush all elements in temporary buffer to memory if an episode is done
                self.merge(self.tb, self.tb_capacity if self.tb_full else self.tb_idx)
                self.tb_full = False
                self.tb_idx = 0
            elif self.tb_full:
                # add the ready experience in temporary buffer to memory
                self.merge(self.tb, 1, self.tb_idx)
        else:
            with self.locker:
                add_buffer(self.memory, self.exp_id, state, action, reward,
                            next_state, done, self.n_steps, self.gamma)
                self.exp_id += 1

    def _sample(self):
        raise NotImplementedError

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

    def _get_samples(self, indexes):
        indexes = list(indexes) # convert tuple to list

        state = self.memory['state'][indexes] 
        next_state = self.memory['next_state'][indexes]

        return (
            state,
            self.memory['action'][indexes],
            self.memory['reward'][indexes],
            next_state,
            self.memory['done'][indexes],
            self.memory['steps'][indexes],
        )
