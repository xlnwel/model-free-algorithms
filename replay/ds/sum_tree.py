import numpy as np
from replay.ds.container import Container

class SumTree(Container):
    """ Interface """
    def __init__(self, capacity):
        super().__init__(capacity)
        self.prio_id = 0
        self.full = False
        # expect the first capacity - 1 elements in self.container are of type np.array([])
        # others are self.prio_expid if data has been filled in
        self.container = [np.array([0.]) for _ in range(2 * capacity - 1)]

    def __len__(self):
        return self.capacity if self.full else self.prio_id

    @property
    def total_priorities(self):
        return self.container[0][0]

    def find(self, value, i, high, total, total2):
        idx = 0                 # start from the root
        if value > self.total_priorities:
            print()
            print('value:', value)
            print('i, high, total, total2:', i, high, total, total2, sep='\t')
            print('total_priority:', self.total_priorities)
            print('segment:', self.total_priorities / 512)
            print('last segment:', self.total_priorities / 512 * 512)

        while idx < self.capacity - 1:
            left, right = 2 * idx + 1, 2 * idx + 2
            if value <= self.container[left][0]:
                idx = left
            else:
                idx = right
                value -= self.container[left][0]
            # if self.container[idx][0] == 0:
            #     print('find value:', value)
            #     print('internal idx:', idx)
            #     print('internal sibling:', left, self.container[left], right, self.container[right])

        # if not isinstance(self.container[idx], self.prio_expid):
        #     print('find value:', value)
        #     print('idx:', idx)
        #     print('item:', self.container[idx])
        #     par = (idx-1) // 2
        #     while self.container[par] == 0:
        #         print('parrent:', par, self.container[par])
        #         par = (par-1) // 2
        #     print('parent:', par, self.container[par])
        #     print('sibling:', 2*par + 1, self.container[2*par+1], 2*par+2, self.container[2*par+2])

        return self.container[idx]

    """ Implementation """
    def _update(self, priority, exp_id):
        prio_id = self.exp2prio[exp_id]
        idx = prio_id + self.capacity - 1
        self.container[idx] = self.prio_expid(priority, exp_id)

        self._propagate(idx)

    def _add(self, priority, exp_id):
        self._update_id_dicts(self.prio_id, exp_id)
        self.prio_id += 1

        if self.prio_id == self.capacity:
            self.full = True
            self.prio_id = 0

        self._update(priority, exp_id)

    def _propagate(self, idx):
        while idx > 0:
            idx = (idx - 1) // 2    # update idx to its parent idx

            left = idx * 2 + 1
            right = idx * 2 + 2

            self.container[idx][0] = self.container[left][0] + self.container[right][0]
