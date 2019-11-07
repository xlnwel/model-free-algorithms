import numpy as np
from copy import deepcopy

from utility.display import assert_colorize, pwc
from utility.utils import moments, standardize, convert_indices


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
            raise NotImplementedError("code has been modified and no longer supports random read for non-lstm networks")
        assert_colorize(epslen // n_minibatches * n_minibatches == epslen, 
            f'Epslen{epslen} is not divisible by #minibatches{n_minibatches}')
        
        assert_colorize(n_envs // n_minibatches * n_minibatches == n_envs, 
            f'#envs({n_envs}) is not divisible by #minibatches{n_minibatches}')

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
            mask=np.zeros((*self.basic_shape, 1), dtype=np.float32),
        )

        self.reset()

    def add(self, **data):
        assert_colorize(self.idx < self.epslen, 
            f'Out-of-range idx {self.idx}. Call "self.reset" beforehand')
        idx = self.idx

        for k, v in data.items():
            if v is not None:
                self[k][:, idx] = v

        self.idx += 1
        # cond = (np.reshape(data['nonterminal'], -1) == 0) * (self['traj_len'] == 0)
        # self['traj_len'] = np.where(cond, self.idx, self['traj_len'])

    def get_batch(self):
        assert_colorize(self.ready, f'PPOBuffer is not ready to be read. Call "self.finish" first')
        start = self.batch_idx * self.minibatch_size
        end = (self.batch_idx + 1) * self.minibatch_size
        self.batch_idx = (self.batch_idx + 1) % self.n_minibatches

        keys = ['state', 'action', 'traj_ret', 'value', 'advantage', 'old_logpi', 'mask']
        if self.use_lstm:
            return {k: self[k][self.indices[start:end], :self.idx].reshape((self.minibatch_size * self.idx, *self[k].shape[2:])) for k in keys}
        else:
            indices = convert_indices(self.indices[start:end], *self.basic_shape)
            return {k: self[k][indices].reshape((self.minibatch_size, *self[k].shape[2:])) for k in keys}
        
    def finish(self, last_value, adv_type, gamma, gae_discount):
        self['value'][:, self.idx] = last_value
        self['mask'][:, self.idx:] = 0
        valid_slice = np.s_[:, :self.idx]
        mask = self['mask'][valid_slice]

        if adv_type == 'nae':
            traj_ret = self['traj_ret'][valid_slice]
            next_return = last_value
            for i in reversed(range(self.idx)):
                traj_ret[:, i] = next_return = self['reward'][:, i] + self['nonterminal'][:, i] * gamma * next_return

            # standardize traj_ret and advantages
            traj_ret_mean, traj_ret_std = moments(traj_ret, mask=mask)
            value = standardize(self['value'][valid_slice], mask=mask)
            value = (value + traj_ret_mean) / (traj_ret_std + 1e-8)     # to have the same mean and std as trajectory return
            self['advantage'][valid_slice] = standardize(traj_ret - value, mask=mask)
            self['traj_ret'][valid_slice] = standardize(traj_ret, mask=mask)
        elif adv_type == 'gae':
            advs = delta = (self['reward'][valid_slice] 
                            + self['nonterminal'][valid_slice] * gamma * self['value'][:, 1:self.idx+1]
                            - self['value'][valid_slice])
            next_adv = 0
            for i in reversed(range(self.idx)):
                advs[:, i] = next_adv = delta[:, i] + self['nonterminal'][:, i] * gae_discount * next_adv
            self['traj_ret'][valid_slice] = advs + self['value'][valid_slice]
            self['advantage'][valid_slice] = standardize(advs, mask=mask)
        else:
            raise NotImplementedError(f'Advantage type should be either "nae" or "gae", but get "{adv_type}".')

        for k, v in self.items():
            v[valid_slice] = (v[valid_slice].T * mask.T).T
        
        self.ready = True

    def reset(self):
        self.idx = 0
        self.batch_idx = 0
        self.ready = False      # whether the buffer is ready to be read

