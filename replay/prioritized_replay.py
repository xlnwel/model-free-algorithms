import threading
import numpy as np
import ray

from replay.utils import add_buffer, copy_buffer


class PrioritizedReplay():
    """ Interface """
    def __init__(self, args, state_dim, action_dim):
        self.memory = {}                        # exp_id    -->     exp
        self.data_structure = None              # prio_id   -->     priority, exp_id
        
        # params for general replay buffer
        self.capacity = int(float(args['capacity']))
        self.min_size = int(float(args['min_size']))
        self.batch_size = args['batch_size']

        # params for prioritized replay
        self.alpha = float(args['alpha']) if 'alpha' in args else .5
        self.beta = float(args['beta0']) if 'beta0' in args else .4
        self.epsilon = float(args['epsilon']) if 'epsilon' in args else 1e-4
        self.beta_grad = (1 - self.beta) / float(args['beta_steps']) * 100

        self.n_steps = args['n_steps']
        self.gamma = args['gamma']
        self.top_priority = 2.

        self.is_full = False
        self.exp_id = 0
        self.sample_i = 0   # count how many times self.sample is called

        # locker used to avoid conflict introduced by tf.data.Dataset and multi-agent
        self.locker = threading.Lock()

        def init_buffer(buffer, capacity, state_dim, action_dim, has_priority):
            target_buffer = {'priority': np.zeros((capacity, 1))} if has_priority else {}
            target_buffer.update({
                'state': np.zeros((capacity, state_dim)),
                'action': np.zeros((capacity, action_dim)),
                'reward': np.zeros((capacity, 1)),
                'next_state': np.zeros((capacity, state_dim)),
                'done': np.zeros((capacity, 1)),
                'steps': np.zeros((capacity, 1))
            })

            buffer.update(target_buffer)

        init_buffer(self.memory, self.capacity, state_dim, action_dim, False)

        # Code for single agent
        if self.n_steps > 1:
            self.temporary_buffer = {}
            init_buffer(self.temporary_buffer, self.n_steps, state_dim, action_dim, True)
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

    def sample(self):
        self.locker.acquire()
        
        samples = self._sample()
        if self.sample_i % 100:
            self._update_beta()

        self.locker.release()

        return samples

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
        self.locker.acquire()
        for priority, exp_id in zip(priorities, saved_exp_ids):
            self.data_structure.update(priority, exp_id)
        self.locker.release()

    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = np.power(probabilities * N, -self.beta)
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upwards

        return IS_ratios
        
    """ Implementation """
    def _sample(self):
        raise NotImplementedError

    def _update_beta(self):
        self.beta = min(self.beta + self.beta_grad, 1)

    # Code for single agent
    def add(self, state, action, reward, next_state, done):
        if self.n_steps > 1:
            add_buffer(self.temporary_buffer, self.tb_idx, state, action, reward, 
                        next_state, done, self.n_steps, self.gamma)
            self.temporary_buffer['priority'][self.tb_idx] = self.top_priority
            
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
