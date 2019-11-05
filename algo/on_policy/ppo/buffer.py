import numpy as np
from copy import deepcopy

from utility.display import assert_colorize, pwc
from utility.utils import standardize, convert_indices


class PPOBuffer(dict):
    def __init__(self, n_envs, epslen, n_minibatches, state_shape, state_dtype, action_shape, action_dtype, mask, use_lstm):
        self.n_envs = n_envs
        self.epslen = epslen
        self.n_minibatches = n_minibatches
        self.use_lstm = use_lstm
        if use_lstm:
            self.indices = np.arange(self.n_envs)
            self.minibatch_size = (self.n_envs) // self.n_minibatches
        else:
            self.indices = np.arange(n_envs * epslen)
            self.minibatch_size = (self.n_envs * self.epslen) // self.n_minibatches
        assert_colorize(epslen // n_minibatches * n_minibatches == epslen, 
            f'Epslen{epslen} is not divisible by #minibatches{n_minibatches}')

        self.basic_shape = (n_envs, epslen)
        super().__init__(
            state=np.zeros((*self.basic_shape, *state_shape), dtype=state_dtype),
            action=np.zeros((*self.basic_shape, *action_shape), dtype=action_dtype),
            reward=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            nonterminal=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            value=np.zeros((n_envs, epslen+1, 1), dtype=np.float32),
            traj_ret=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            advantage=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            old_logpi=np.zeros((*self.basic_shape, 1), dtype=np.float32),
            mask=np.zeros((*self.basic_shape, 1), dtype=np.float32) if mask else None
        )

        self.reset()

    def add(self, **data):
        assert_colorize(self.idx < self.epslen, 
            f'Out-of-range idx {self.idx}. Call self.reset() beforehand')
        idx = self.idx

        for k, v in data.items():
            if v is not None:
                self[k][:, idx] = v

        self.idx += 1

    def get_batch(self):
        if self.batch_idx == 0:
            self._shuffle()
        start = self.batch_idx * self.minibatch_size
        end = (self.batch_idx + 1) * self.minibatch_size
        self.batch_idx = (self.batch_idx + 1) % self.n_minibatches

        keys = ['state', 'action', 'traj_ret', 'value', 'advantage', 'old_logpi', 'mask']
        if self.use_lstm:
            return {k: self[k][self.indices[start:end], :self.epslen].reshape((self.minibatch_size * self.epslen, *self[k].shape[2:])) for k in keys}
        else:
            indices = convert_indices(self.indices[start:end], *self.basic_shape)
            return {k: self[k][indices].reshape((self.minibatch_size, *self[k].shape[2:])) for k in keys}
        
    def compute_ret_adv(self, last_value, adv_type, gamma, gae_discount):
        self['value'][:, -1] = last_value
        mask = self.get('mask')

        if adv_type == 'nae':
            traj_ret = self['traj_ret']
            next_return = 0
            for i in reversed(range(self.epslen)):
                traj_ret[:, i] = next_return = self['reward'][:, i] + self['nonterminal'][:, i] * gamma * next_return

            # standardize traj_ret and advantages
            values = standardize(self['value'][:-1], mean=np.mean(traj_ret), std=np.std(traj_ret), mask=mask)
            self['advantage'] = standardize(traj_ret - values, mask=mask)
            self['traj_ret'] = standardize(traj_ret, mask=mask)
        elif adv_type == 'gae':
            advs = delta = self['reward'] + self['nonterminal'] * gamma * self['value'][:, 1:] - self['value'][:, :-1]
            # advs = np.zeros_like(delta)
            next_adv = 0
            for i in reversed(range(self.epslen)):
                advs[:, i] = next_adv = delta[:, i] + self['nonterminal'][:, i] * gae_discount * next_adv
            self['traj_ret'] = advs + self['value'][:, :-1]
            self['advantage'] = standardize(advs, mask=mask)
        else:
            NotImplementedError(f'Advantage type should be either "nae" or "gae".')

        if mask is not None:
            for k, v in self.items():
                self[k] = (v[:, :self.epslen].T * mask.T).T

    def reset(self):
        self.idx = 0
        # restore value, which is corrupted at the end of compute_ret_adv
        self['value'] = np.zeros((self.n_envs, self.epslen+1, 1), dtype=np.float32)
        self.batch_idx = 0

    def _shuffle(self):
        assert_colorize(self.batch_idx == 0, f'Erroneous shuffle timing: batch index is {self.batch_idx} not zero')
        np.random.shuffle(self.indices)
