import numpy as np
from replay.replaybuffer import ReplayBuffer

class PrioritizedReplay(ReplayBuffer):
    """ Interface """
    def __init__(self, args, sample_size, n_steps=1, gamma=1):
        super().__init__(args, sample_size, n_steps=n_steps, gamma=gamma)
        self.memory = []                        # exp_id    -->     exp
        self.data_structure = None              # prio_id   -->     priority, exp_id
        
        self._alpha = args['alpha'] if 'alpha' in args else .5
        self._beta = args['beta0'] if 'beta0' in args else .4
        self._epsilon = args['epsilon'] if 'epsilon' in args else 1e-4
        self._beta_grad = (1 - self._beta) / args['beta_steps'] if 'beta_steps' in args else 1e-4

        self._exp_id = -1
        self._saved_exp_ids = None               # saved exp ids, whose priorities will get updated latter
        self.beta1 = False
        self.top_priority = args['top_priority'] if 'top_priority' in args else 2.

    def sample(self):
        assert self.good_to_learn, 'There are not sufficient transitions in buffer to learn'
        IS_ratios, self._saved_exp_ids, exps = self._sample()

        return IS_ratios, self._unpack_samples(exps)

    def update_priorities(self, priorities):
        priorities = self._compute_priority(priorities)
        
        if self._saved_exp_ids is None:
            raise ValueError('update_priorities() should be called only after sample()!')
        
        for priority, exp_id in zip(priorities, self._saved_exp_ids):
            if priority > self.top_priority:
                self.top_priority = priority
            self.data_structure.update(priority, exp_id)

        self._saved_exp_ids = None
        
    def __len__(self):
        return len(self.memory)

    """ Implementation """
    def _sample(self):
        """ return importance sampling ratios, saved exp ids, exps """
        raise NotImplementedError

    def _add_exp(self, exp):
        """ 
        add exp to self.memory
        in this process, we apply FIFO when the memory is full
        """
        self._exp_id += 1
        if self._exp_id >= self.capacity:
            self._exp_id = 0
        if self._exp_id < len(self):
            self.memory[self._exp_id] = exp
        else:
            self.memory.append(exp)

        priority = self.top_priority
        self.data_structure.update(priority, self._exp_id)

    def _update_beta(self):
        self._beta = min(self._beta + self._beta_grad, 1)
        
        # test when beta reaches 1
        if not self.beta1 and self._beta == 1:
            print('\nbeta == 1')
            self.beta1 = True

    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = np.power(probabilities * N, -self._beta)
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upwards

        return IS_ratios

    def _compute_priority(self, priorities):
        raise NotImplementedError
