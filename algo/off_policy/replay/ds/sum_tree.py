import numpy as np
from algo.off_policy.replay.ds.container import Container

class SumTree(Container):
    """ Interface """
    def __init__(self, capacity):
        super().__init__(capacity)
        self.prio_id = 0
        self.full = False
        # expect the first capacity - 1 elements in self.container are of type np.array([])
        # others are self.prio_expid if data has been filled in
        self.container = np.zeros((2 * capacity - 1))

    @property
    def total_priorities(self):
        return self.container[0]

    def find(self, value):
        idx = 0                 # start from the root

        while idx < self.capacity - 1:
            left, right = 2 * idx + 1, 2 * idx + 2
            if value <= self.container[left]:
                idx = left
            else:
                idx = right
                value -= self.container[left]

        return self.container[idx], idx - self.capacity + 1

    def add(self, priority, exp_id, full):        
        self.update(priority, exp_id)
        assert exp_id == self.prio_id, f'{exp_id} != {self.prio_id}'
        self.prio_id = (self.prio_id + 1) % self.capacity
        self.full = full

    def update(self, priority, exp_id):
        idx = exp_id + self.capacity - 1
        self.container[idx] = priority

        self._propagate(idx)

    def _propagate(self, idx):
        while idx > 0:
            idx = (idx - 1) // 2    # update idx to its parent idx

            left = idx * 2 + 1
            right = idx * 2 + 2

            self.container[idx] = self.container[left] + self.container[right]
