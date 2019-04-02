import numpy as np

from utility.utils import normalize


class PPOBuffer(dict):
    def __init__(self, n_envs, seq_len, n_minibatches, minibatch_size, state_space, action_dim):
        assert seq_len // n_minibatches == minibatch_size
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.minibatch_size = minibatch_size

        self.indices = np.arange(seq_len)
        self.idx = 0

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

        result = np.reshape(self[key][:, self.indices[start: end]], (self.n_envs * self.minibatch_size, -1))

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

    def reset(self):
        self.idx = 0

    def normalize_adv(self, mean, std):
        self['advantage'] = (self['advantage'] - mean) / (std + 1e8)

    def shuffle(self):
        np.random.shuffle(self.indices)