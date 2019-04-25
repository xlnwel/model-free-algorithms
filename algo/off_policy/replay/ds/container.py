from collections import namedtuple

class Container():
    """ Interface """
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []
        self.prio2exp = {}
        self.exp2prio = {}
        self.prio_expid = namedtuple('prio_expid', ('priority', 'exp_id'))

    def update(self, priority, exp_id):
        """ we don't check if there'll be too many items in self.container.
        This is guaranteed by Prioritized Replay via reusing exp_id
        """
        if exp_id in self.exp2prio:
            # exp_id is already in the heap, replace it
            self._update(priority, exp_id)
        else:
            self._add(priority, exp_id)

    def __len__(self):
        raise NotImplementedError

    """ Implementation """
    def _update(self, priority, exp_id):
        """ update data structure """
        raise NotImplementedError

    def _add(self, priority, exp_id):
        """ add item to data structure """
        raise NotImplementedError

    def _update_id_dicts(self, prio_id, exp_id=None):
        if exp_id is None:
            exp_id = self.container[prio_id].exp_id
        self.prio2exp[prio_id] = exp_id
        self.exp2prio[exp_id] = prio_id
        