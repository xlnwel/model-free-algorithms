import threading
import numpy as np

from utility.decorators import override
from replay.basic_replay import Replay
from replay.utils import add_buffer, copy_buffer


class PrioritizedReplay(Replay):
    """ Interface """
    def __init__(self, args, state_space, action_dim):
        super().__init__(args, state_space, action_dim)
        # self.memory                           # exp_id    -->     exp
        self.data_structure = None              # prio_id   -->     priority, exp_id

        # params for prioritized replay
        self.alpha = float(args['alpha']) if 'alpha' in args else .5
        self.beta = float(args['beta0']) if 'beta0' in args else .4
        self.epsilon = float(args['epsilon']) if 'epsilon' in args else 1e-4
        self.beta_grad = (1 - self.beta) / float(args['beta_steps']) * 100

        self.top_priority = 2.

        self.sample_i = 0   # count how many times self.sample is called

        # locker used to avoid conflict introduced by tf.data.Dataset and multi-agent
        self.locker = threading.Lock()

    @override(Replay)
    def sample(self):
        with self.locker:        
            samples = self._sample()
            if self.sample_i % 100:
                self._update_beta()

        return samples

    @override(Replay)
    def merge(self, local_buffer, length, start=0):
        assert length < self.capacity, 'Local buffer is too large'
        self.locker.acquire()

        end_idx = self.exp_id + length

        if end_idx > self.capacity:
            first_part = self.capacity - self.exp_id
            second_part = length - first_part
            
            copy_buffer(self.memory, self.exp_id, self.capacity, local_buffer, start, start + first_part)
            copy_buffer(self.memory, 0, second_part, local_buffer, start + first_part, start + length)
        else:
            copy_buffer(self.memory, self.exp_id, end_idx, local_buffer, start, start + length)

        for prio_id, exp_id in enumerate(range(self.exp_id, end_idx)):
            self.data_structure.update(local_buffer['priority'][prio_id], exp_id % self.capacity)
        
        # memory is full, recycle buffer via FIFO
        if not self.is_full and end_idx >= self.capacity:
            print('Memory is fulll')
            self.is_full = True
        self.exp_id = end_idx % self.capacity

        self.locker.release()

    def update_priorities(self, priorities, saved_exp_ids):
        with self.locker:
            for priority, exp_id in zip(priorities, saved_exp_ids):
                self.data_structure.update(priority, exp_id)

    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = np.power(probabilities * N, -self.beta)
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upwards

        return IS_ratios
    
    # Code for single agent
    def add(self, state, action, reward, next_state, done):
        if self.n_steps > 1:
            self.temporary_buffer['priority'][self.tb_idx] = self.top_priority
        super().add(state, action, reward, next_state, done)

    """ Implementation """
    def _sample(self):
        raise NotImplementedError

    def _update_beta(self):
        self.beta = min(self.beta + self.beta_grad, 1)
