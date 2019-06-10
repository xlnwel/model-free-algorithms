from collections import namedtuple

class Container():
    """ Interface """
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []

    def add(self, priority, mem_idx, full):
        raise NotImplementedError

    def update(self, priority, mem_idx):
        raise NotImplementedError

    def find(self, value):
        raise NotImplementedError
