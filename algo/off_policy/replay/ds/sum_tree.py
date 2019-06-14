import numpy as np
from algo.off_policy.replay.ds.container import Container

class SumTree(Container):
    """ Interface """
    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree_size = capacity - 1
        
        self.container = np.zeros(self.tree_size + self.capacity)

    @property
    def total_priorities(self):
        return self.container[0]

    def find(self, value):
        idx = 0                 # start from the root

        while idx < self.tree_size:
            left, right = 2 * idx + 1, 2 * idx + 2
            if value <= self.container[left]:
                idx = left
            else:
                idx = right
                value -= self.container[left]

        return self.container[idx], idx - self.tree_size

    def update(self, priority, mem_idx):
        idx = mem_idx + self.tree_size
        self.container[idx] = priority

        self._propagate(idx)

    def _propagate(self, idx):
        while idx > 0:
            idx = (idx - 1) // 2    # update idx to its parent idx

            left = idx * 2 + 1
            right = idx * 2 + 2

            self.container[idx] = self.container[left] + self.container[right]
