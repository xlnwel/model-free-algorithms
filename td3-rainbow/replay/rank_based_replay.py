import numpy as np
from replay.ds.priotity_queue import PriorityQueue
from replay.prioritized_replay import PrioritizedReplay

class RankBasedPrioritizedReplay(PrioritizedReplay):
    """ Interface """
    def __init__(self, args, sample_size, n_steps=1, gamma=1):
        super().__init__(args, sample_size, n_steps=n_steps, gamma=gamma)
        self.data_structure = PriorityQueue(self.capacity)

        self.total_categories = args['categories'] if 'categories' in args else 2
        self.min_elements = args['min_elements'] if 'min_elements' in args else 1
        self.categorical_ranks = self._build_ranks()

    """ Implementation """
    def _build_ranks(self):
        categorical_ranks = []

        # build categories
        for category_max in np.linspace(self.min_size, self.capacity, self.total_categories):
            n = int(category_max)
            assert self.sample_size * self.min_elements < n, 'The minimum category size is not satisfied'
            
            # P(i) = i ^ (-alpha) / sum(k ^ (-alpha))
            pdf = np.array([np.power(x, -self._alpha) for x in range(1, n+1)])
            pdf_sum = np.sum(pdf)
            pdf = pdf / pdf_sum

            # compute the range for each of sample_size segments
            cdf = np.cumsum(pdf)
            segment_size = 1. / self.sample_size
            segments = []
            idx = 0

            for seg in range(self.sample_size):
                start = idx
                idx += self.min_elements    # ensure at least self.min_elements in a segment
                while idx < n and cdf[idx] < (seg + 1) * segment_size:
                    idx += 1
                end = idx
                segments.append((start, end))
            segments[-1] = (start, n)   # avoid loss of precision caused by float point

            categorical_ranks.append(segments)

        print('{} Categorical Ranks:'.format(len(categorical_ranks)))
        for i, rank in enumerate(categorical_ranks):
            print('rank {}:'.format(i), rank, seg='\n')
        return categorical_ranks

    def _sample(self):
        """ return importance sampling ratios, saved exp ids, exps """
        category = max(self.total_categories * len(self) // self.capacity - 1, 0)
        ranks = self.categorical_ranks[category]
        # priority IDs
        prio_ids = [np.random.randint(start, end) for start, end in ranks]
        
        self._update_beta()

        # compute importance sampling ratios
        N = ranks[-1][1]
        probabilities = np.array([1 / ((end - start) * self.sample_size) for start, end in ranks])
        IS_ratios = self._compute_IS_ratios(N, probabilities)

        exp_ids = self.data_structure.get_exp_ids(prio_ids)
        
        return IS_ratios, exp_ids, [self.memory[i] for i in exp_ids]
        
    def _compute_priority(self, priorities):
        return priorities
