import numpy as np

from utility.utils import normalize, assert_colorize


class PPOBuffer(dict):
    def __init__(self, n_envs, seq_len, n_minibatches, state_space, action_dim, mask, use_rnn):
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.n_minibatches = n_minibatches
        # self.minibatch_size = n_envs // n_minibatches if use_rnn else seq_len // n_minibatches
        self.minibatch_size = seq_len // n_minibatches
        self.mask = mask

        self.idx = 0
        
        # self.use_rnn = use_rnn
        # index environment dimension when using rnn, sequence dimension when not
        # self.indices = np.arange(n_envs) if use_rnn else np.arange(seq_len)
        self.indices = np.arange(seq_len)

        basic_shape = (n_envs, seq_len)
        super().__init__({
            'state': np.zeros((*basic_shape, *state_space)),
            'action': np.zeros((*basic_shape, action_dim)),
            'reward': np.zeros(basic_shape),
            'nonterminal': np.zeros(basic_shape),
            'value': np.zeros((n_envs, seq_len + 1)),
            'return': np.zeros(basic_shape),
            'advantage': np.zeros(basic_shape),
            'old_logpi': np.zeros(basic_shape)
        })

    def add(self, state, action, reward, value, logpi, nonterminal):
        assert_colorize(self.idx < self.seq_len, f'Out-of-range idx {self.idx}. Call self.reset() beforehand')
        idx = self.idx
        self['state'][:, idx] = state
        self['action'][:, idx] = action
        self['reward'][:, idx] = reward
        self['value'][:, idx] = value
        self['old_logpi'][:, idx] = logpi
        self['nonterminal'][:, idx] = nonterminal
        self.idx += 1

    def get_flat_batch(self, key, batch_idx):
        start = batch_idx * self.minibatch_size
        end = (batch_idx + 1) * self.minibatch_size
        
        shape = (self.n_envs * self.seq_len // self.n_minibatches, *(self[key].shape[2:] if len(self[key].shape) > 2 else (-1, )))
        # result = (self[key][self.indices[start: end], :self.seq_len] if self.use_rnn else self[key][:, self.indices[start: end]]).reshape(shape)
        result = self[key][:, self.indices[start: end]].reshape(shape)

        return result

    def compute_ret_adv(self, adv_type, gamma, gae_discount):
        if adv_type == 'norm':
            returns = self['return']
            next_return = 0
            for i in reversed(range(self.seq_len)):
                returns[:, i] = next_return = self['reward'][:, i] + self['nonterminal'][:, i] * gamma * next_return

            # normalize returns and advantages
            values = normalize(self['value'][:, :-1], np.mean(returns), np.std(returns))
            self['advantage'] = normalize(returns - values)
            self['return'] = normalize(returns)
        elif adv_type == 'gae':
            advs = delta = self['reward'] + self['nonterminal'] * gamma * self['value'][:, 1:] - self['value'][:, :-1]
            next_adv = 0
            for i in reversed(range(self.seq_len)):
                advs[:, i] = next_adv = delta[:, i] + self['nonterminal'][:, i] * gae_discount * next_adv
            self['return'] = advs + self['value'][:, :-1]
            self['advantage'] = advs
        else:
            NotImplementedError

        if self.mask:
            for k in self.keys():
                shape = list(self['nonterminal'].shape) + [1] * (len(self[k].shape) - len(self['nonterminal'].shape))
                mask = self['nonterminal'].reshape(shape)
                self[k][:, :self.seq_len] * mask

    def reset(self):
        self.idx = 0

    def normalize_adv(self, mean, std):
        self['advantage'] = (self['advantage'] - mean) / (std + 1e8)

    def shuffle(self):
        np.random.shuffle(self.indices)