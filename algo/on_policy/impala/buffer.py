import time
from threading import Lock
import numpy as np

from utility.utils import assert_colorize, pwc

class IMPALABuffer(dict):
    def __init__(self, max_size, seq_len, state_space, action_dim):
        self.max_size = max_size
        self.idx = 0
        self.size = 0
        self.locker = Lock()

        basic_shape = (max_size, seq_len)
        super().__init__({
            'state': np.zeros((*basic_shape, *state_space)),
            'action': np.zeros((*basic_shape, action_dim)),
            'reward': np.zeros(basic_shape),
            'nonterminal': np.zeros(basic_shape),
            'pi': np.zeros((basic_shape))
        })

    def get(self, n):
        assert_colorize(n <= self.size)

        keys = ['state', 'action', 'reward', 'nonterminal', 'pi', 'lstm_state']
        with self.locker:
            range = np.arange(self.idx, self.idx + n) % self.max_size
            data = [self[key][range] for key in keys]
            self.idx = (self.idx + n) % self.max_size
            self.size -= n

        return data

    def put(self, state, action, reward, nonterminal, pi, lstm_state):
        while self.size >= self.max_size:
            pwc('full buffer')
            time.sleep(.1)

        with self.locker:
            idx = self.idx
            self['state'][idx] = state
            self['action'][idx] = action
            self['reward'][idx] = reward
            self['nonterminal'][idx] = nonterminal
            self['pi'][idx] = pi
            self['lstm_state'][idx] = lstm_state
            self.idx = (self.idx + 1) % self.max_size
