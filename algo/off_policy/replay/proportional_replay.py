import numpy as np
import ray

from utility.decorators import override
from algo.off_policy.replay.ds.sum_tree import SumTree
from algo.off_policy.replay.prioritized_replay import PrioritizedReplay


class ProportionalPrioritizedReplay(PrioritizedReplay):
    """ Interface """
    def __init__(self, args, state_space, action_dim):
        super().__init__(args, state_space, action_dim)
        self.data_structure = SumTree(self.capacity)        # mem_idx    -->     priority

    """ Implementation """
    @override(PrioritizedReplay)
    def _sample(self):
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        priorities, indexes = list(zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment))
                                        for i in range(self.batch_size)]))

        priorities = np.array(priorities)
        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        IS_ratios = self._compute_IS_ratios(probabilities)
        samples = self._get_samples(indexes)
        
        return IS_ratios, indexes, samples
