import numpy as np

from utility.decorators import override
from utility.utils import assert_colorize
from algo.off_policy.replay.basic_replay import Replay
from algo.off_policy.replay.utils import add_buffer, copy_buffer


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

    @override(Replay)
    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:        
            samples = self._sample()
            self.sample_i += 1
            if self.sample_i % 100 == 0:
                self._update_beta()

        return samples

    @override(Replay)
    def add(self, state, action, reward, next_state, done):
        if self.n_steps > 1:
            self.temporary_buffer['priority'][self.tb_idx] = self.top_priority
        super()._add(state, action, reward, next_state, done)

    def update_priorities(self, priorities, saved_exp_ids):
        with self.locker:
            for priority, exp_id in zip(priorities, saved_exp_ids):
                self.data_structure.update(priority, exp_id)

    """ Implementation """
    def _update_beta(self):
        self.beta = min(self.beta + self.beta_grad, 1)

    @override(Replay)
    def _merge(self, local_buffer, length, start=0):
        end_idx = self.exp_id + length
        for prio_id, exp_id in enumerate(range(self.exp_id, end_idx)):
            self.data_structure.update(local_buffer['priority'][prio_id], exp_id % self.capacity)
            
        super()._merge(local_buffer, length, start)
        
    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = np.power(probabilities * N, -self.beta)
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upward

        return IS_ratios