from collections import namedtuple
import numpy as np
from replay.ds.container import Container

class PriorityQueue(Container):
    """ Interface """
    def __init__(self, capacity):
        super().__init__(capacity)

        # count how many times priorities are changed
        # sort container when steps reaches capacity
        self.steps = 0
        
    def update(self, priority, exp_id):
        """ we don't check if there'll be too many items in self.container.
        This is guaranteed by Prioritized Replay via reusing exp_id
        """
        super().update(priority, exp_id)

        self._update_steps()

    def __len__(self):
        return len(self.container)
        
    """ Implementation """
    def _update(self, priority, exp_id):
        prio_id = self.exp2prio[exp_id]
        self.container[prio_id] = self.prio_expid(priority, exp_id)
        self.prio2exp[prio_id] = exp_id

        self._upheap(prio_id)
        self._downheap(prio_id)

    def _add(self, priority, exp_id):
        prio_id = len(self.container)
        self.container.append(self.prio_expid(priority, exp_id))
        self._update_id_dicts(prio_id, exp_id)

        self._upheap(prio_id)

    def _upheap(self, idx):        
        while idx != 0:
            parent = (idx - 1) // 2

            if self._get_priority(parent) >= self._get_priority(idx):
                break
                
            self._swap(parent, idx)

            idx = parent

    def _downheap(self, idx):
        while 2 * idx + 1 < len(self):
            left = 2 * idx + 1
            right = left + 1 if left + 1 < len(self) else left

            greatest_idx = np.argmax([self._get_priority(idx), self._get_priority(left), self._get_priority(right)])
            
            if greatest_idx == 0:
                break

            swapped_idx = left if greatest_idx == 1 else right
            self._swap(idx, swapped_idx)
            
            idx = swapped_idx

    def _swap(self, idx1, idx2):
        self.container[idx1], self.container[idx2] = self.container[idx2], self.container[idx1]
            
        self._update_id_dicts(idx1)
        self._update_id_dicts(idx2)

    def _get_priority(self, prio_id):
        return self.container[prio_id].priority

    def _update_steps(self):
        self.steps += 1
        if self.steps >= self.capacity * 10:
            self.container.sort(reverse=True)
            self.steps = 0
            print('Priority queue has been resorted')

            # update dicts
            for prio_id, (_, exp_id) in enumerate(self.container):
                self._update_id_dicts(prio_id, exp_id)
