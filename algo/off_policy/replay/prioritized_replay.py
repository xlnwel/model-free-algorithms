import numpy as np

from utility.decorators import override
from utility.debug_tools import assert_colorize
from utility.schedule import PiecewiseSchedule
from algo.off_policy.replay.basic_replay import Replay
from algo.off_policy.replay.utils import init_buffer, add_buffer, copy_buffer


class PrioritizedReplay(Replay):
    """ Interface """
    def __init__(self, args, state_space, action_dim):
        super().__init__(args, state_space, action_dim)
        self.data_structure = None            

        # params for prioritized replay
        self.alpha = float(args['alpha']) if 'alpha' in args else .5
        self.beta = float(args['beta0']) if 'beta0' in args else .4
        self.beta_schedule = PiecewiseSchedule([(0, args['beta0']), (float(args['beta_steps']), 1.)], 
                                                outside_value=1.)
        self.epsilon = float(args['epsilon']) if 'epsilon' in args else 1e-4

        self.top_priority = 2.
        self.to_update_priority = args['to_update_priority'] if 'to_update_priority' in args else True

        self.sample_i = 0   # count how many times self.sample is called

        init_buffer(self.memory, self.capacity, state_space, action_dim, self.n_steps == 1)

        # Code for single agent
        if self.n_steps > 1:
            self.tb_capacity = args['tb_capacity']
            self.tb_idx = 0
            self.tb_full = False
            self.tb = {}
            init_buffer(self.tb, self.tb_capacity, state_space, action_dim, True)

    @override(Replay)
    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:        
            samples = self._sample()
            self.sample_i += 1
            self._update_beta()

        return samples

    @override(Replay)
    def add(self, state, action, reward, done):
        if self.n_steps > 1:
            self.tb['priority'][self.tb_idx] = self.top_priority
        else:
            self.memory['priority'][self.mem_idx] = self.top_priority
            self.data_structure.update(self.top_priority, self.mem_idx)
        super()._add(state, action, reward, done)

    def update_priorities(self, priorities, saved_mem_idxs):
        with self.locker:
            if self.to_update_priority:
                self.top_priority = max(self.top_priority, np.max(priorities))
            for priority, mem_idx in zip(priorities, saved_mem_idxs):
                self.data_structure.update(priority, mem_idx)

    """ Implementation """
    def _update_beta(self):
        self.beta = self.beta_schedule.value(self.sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):
        end_idx = self.mem_idx + length
        assert np.all(local_buffer['priority'][: length])
        for idx, mem_idx in enumerate(range(self.mem_idx, end_idx)):
            self.data_structure.update(local_buffer['priority'][idx], mem_idx % self.capacity)
            
        super()._merge(local_buffer, length)
        
    def _compute_IS_ratios(self, probabilities):
        IS_ratios = (np.min(probabilities) / probabilities)**self.beta

        return IS_ratios
