import numpy as np
import ray

from utility.decorators import override
from replay.ds.sum_tree import SumTree
from replay.prioritized_replay import PrioritizedReplay


class ProportionalPrioritizedReplay(PrioritizedReplay):
    """ Interface """
    def __init__(self, args, state_space, action_dim):
        super().__init__(args, state_space, action_dim)
        self.data_structure = SumTree(self.capacity)                               # prio_id   -->     priority, exp_id

    """ Implementation """
    @override(PrioritizedReplay)
    def _sample(self):
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        priorities, exp_ids = list(zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment), i, (i+1) * segment, total_priorities, self.data_structure.total_priorities)
                                        for i in range(self.batch_size)]))

        priorities = np.squeeze(priorities)
        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        N = len(self)
        IS_ratios = self._compute_IS_ratios(N, probabilities)
        
        return IS_ratios, exp_ids, self._get_samples(exp_ids)

    def _get_samples(self, exp_ids):
        exp_ids = list(exp_ids) # convert tuple to list

        return (
            self.memory['state'][exp_ids],
            self.memory['action'][exp_ids],
            self.memory['reward'][exp_ids],
            self.memory['next_state'][exp_ids],
            self.memory['done'][exp_ids],
            self.memory['steps'][exp_ids],
        )
