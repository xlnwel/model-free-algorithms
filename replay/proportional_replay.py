import numpy as np

from replay.utils import init_buffer, add_buffer, copy_buffer
from replay.ds.sum_tree import SumTree


class ProportionalPrioritizedReplay():
    """ Interface """
    def __init__(self, args, state_dim, action_dim):
        self.capacity = int(float(args['capacity']))
        self.min_size = int(float(args['min_size']))
        self.batch_size = args['batch_size']
        self.n_steps = args['n_steps']
        self.gamma = args['gamma']

        # params for prioritized replay
        self.alpha = float(args['alpha']) if 'alpha' in args else .5
        self.beta = float(args['beta0']) if 'beta0' in args else .4
        self.epsilon = float(args['epsilon']) if 'epsilon' in args else 1e-4
        self.beta_grad = (1 - self.beta) / float(args['beta_steps'])

        self.memory = init_buffer(self.capacity, state_dim, action_dim, False)      # exp_id    -->     exp
        self.data_structure = SumTree(self.capacity)                                # prio_id   -->     priority, exp_id

        self.exp_id = -1
        self.beta1 = False
        self.top_priority = 2.

        if self.n_steps > 1:
            self.temporary_buffer = init_buffer(self.n_steps, state_dim, action_dim, True)
            self.tb_idx = 0
            self.tb_full = False

    @property
    def good_to_learn(self):
        return len(self) >= self.min_size

    def __len__(self):
        return self.memory['counter']

    def __call__(self):
        while True:
            yield self.sample()

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

    def merge(self, local_buffer, length, start=0):
        assert length < self.capacity, 'Local buffer is too large'
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

        self.exp_id = end_idx % self.capacity

    def sample(self):
        assert self.good_to_learn, 'There are not sufficient transitions in buffer to learn'
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        priorities, exp_ids = list(zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment))
                                    for i in range(self.batch_size)]))

        priorities = np.squeeze(priorities)
        probabilities = priorities / total_priorities

        self._update_beta()
        # compute importance sampling ratios
        N = len(self)
        IS_ratios = self._compute_IS_ratios(N, probabilities)
        
        return IS_ratios, exp_ids, self._get_samples(exp_ids)

    def update_priorities(self, priorities, saved_exp_ids):
        for priority, exp_id in zip(priorities, saved_exp_ids):
            if priority > self.top_priority:
                self.top_priority = priority
            self.data_structure.update(priority, exp_id)
        
    def compute_priorities(self, priorities):
        priorities += self.epsilon
        priorities **= self.alpha
        
        return priorities

    """ Implementation """
    def _sample(self):
        """ return importance sampling ratios, saved exp ids, exps """
        raise NotImplementedError

    def _update_beta(self):
        self.beta = min(self.beta + self.beta_grad, 1)
        
        # test when beta reaches 1
        if not self.beta1 and self.beta == 1:
            print('\nbeta == 1')
            self.beta1 = True

    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = np.power(probabilities * N, -self.beta)
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upwards

        return IS_ratios

    def _get_samples(self, exp_ids):
        exp_ids = list(exp_ids)

        return (
            self.memory['state'][exp_ids],
            self.memory['action'][exp_ids],
            self.memory['reward'][exp_ids],
            self.memory['next_state'][exp_ids],
            self.memory['done'][exp_ids],
            self.memory['steps'][exp_ids],
        )
