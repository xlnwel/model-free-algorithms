import numpy as np

from td3_rainbow.replay.ds.sum_tree import SumTree
from td3_rainbow.replay.prioritized_replay import PrioritizedReplay


class ProportionalPrioritizedReplay(PrioritizedReplay):
    """ Interface """
    def __init__(self, args, sample_size, n_steps=1, gamma=1):
        super().__init__(args, sample_size, n_steps=n_steps, gamma=gamma)
        self.data_structure = SumTree(self.capacity)
        self._epsilon = float(args['epsilon'])

    """ Implementation """
    def _sample(self):
        """ return importance sampling ratios, saved exp ids, exps """
        total_priorities = self.data_structure.total_priorities

        segment = total_priorities / self.sample_size
        
        priorities, exp_ids = zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment)) for i in range(self.sample_size)])

        priorities = np.squeeze(priorities)
        probabilities = priorities / self.data_structure.total_priorities

        self._update_beta()
        # compute importance sampling ratios
        N = len(self)
        IS_ratios = self._compute_IS_ratios(N, probabilities)

        return IS_ratios, exp_ids, [self.memory[i] for i in exp_ids]

    def _compute_priority(self, priorities):
        priorities += self._epsilon
        priorities **= self._alpha
        
        return priorities
