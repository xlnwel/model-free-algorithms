from collections import namedtuple

class Container():
    """ Interface """
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []

    def add(self, priority, exp_id, full):
        raise NotImplementedError

    def update(self, priority, exp_id):
        raise NotImplementedError

    def find(self, value):
        raise NotImplementedError
