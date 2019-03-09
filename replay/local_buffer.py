import numpy as np


class LocalBuffer(dict):
    def __init__(self, length, **kwarg):
        self.fake_ratio = np.zeros(length)
        self.fake_ids = np.zeros(length, dtype=np.int32)

    def __call__(self):
        while True:
            yield self.fake_ratio, self.fake_ids, (self['state'], self['action'], self['reward'], 
                                self['next_state'], self['done'], self['steps'])
