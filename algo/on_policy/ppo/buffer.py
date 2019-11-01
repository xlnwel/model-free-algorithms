import numpy as np
from copy import deepcopy

from utility.debug_tools import assert_colorize
from utility.utils import standardize


class PPOBuffer(dict):
    def __init__(self, n_envs, epslen, n_minibatches, state_space, action_dim, mask):
        self.n_envs = n_envs
        self.epslen = epslen
        self.n_minibatches = n_minibatches
        assert_colorize(epslen // n_minibatches * n_minibatches == epslen, 
            f'Epslen{epslen} is not divisible by #minibatches{n_minibatches}')
        self.minibatch_size = epslen // n_minibatches

        self.idx = 0
        
        self.indices = np.arange(epslen)

        self.basic_shape = (n_envs, epslen)
        super().__init__({
            'state': np.zeros((*self.basic_shape, *state_space)),
            'action': np.zeros((*self.basic_shape, action_dim)),
            'reward': np.zeros(self.basic_shape),
            'nonterminal': np.zeros(self.basic_shape),
            'value': np.zeros((n_envs, epslen + 1)),
            'return': np.zeros(self.basic_shape),
            'advantage': np.zeros(self.basic_shape),
            'old_logpi': np.zeros(self.basic_shape)
        })
        if mask:
            # we use mask in two way, 
            # 1. standardize ret and adv, see self.compute_ret_adv
            # 2. mask loss, see loss function in agent.py
            self['mask'] = np.zeros(self.basic_shape)

    def add(self, **data):
        assert_colorize(self.idx < self.epslen, 
            f'Out-of-range idx {self.idx}. Call self.reset() beforehand')
        idx = self.idx
        for k, v in data.items():
            self[k][:, idx] = v

        self.idx += 1

    def get_batch(self, key, batch_idx):
        start = batch_idx * self.minibatch_size
        end = (batch_idx + 1) * self.minibatch_size
        
        shape = (self.n_envs * self.epslen // self.n_minibatches, 
                * (self[key].shape[2:] if len(self[key].shape) > 2 else (1, )))
        result = self[key][:, self.indices[start:end]].reshape(shape)

        return result

    def compute_ret_adv(self, last_value, adv_type, gamma, gae_discount):
        self['value'][:, -1] = last_value
        mask = self.get('mask')

        if adv_type == 'nae':
            returns = self['return']
            next_return = 0
            for i in reversed(range(self.epslen)):
                returns[:, i] = next_return = self['reward'][:, i] + self['nonterminal'][:, i] * gamma * next_return

            # standardize returns and advantages
            values = standardize(self['value'][:, :-1], mean=np.mean(returns), std=np.std(returns), mask=mask)
            self['advantage'] = standardize(returns - values, mask=mask)
            self['return'] = standardize(returns, mask=mask)
        elif adv_type == 'gae':
            advs = delta = self['reward'] + self['nonterminal'] * gamma * self['value'][:, 1:] - self['value'][:, :-1]
            # advs = np.zeros_like(delta)
            next_adv = 0
            for i in reversed(range(self.epslen)):
                advs[:, i] = next_adv = delta[:, i] + self['nonterminal'][:, i] * gae_discount * next_adv
            self['return'] = advs + self['value'][:, :-1]
            self['advantage'] = standardize(advs, mask=mask)
        else:
            NotImplementedError

    def reset(self):
        self.idx = 0

    def shuffle(self):
        np.random.shuffle(self.indices)
