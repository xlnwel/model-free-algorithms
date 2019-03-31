import threading
from abc import ABCMeta, abstractclassmethod


class Replay(metaclass=ABCMeta):
    def __init__(self, args, state_space, action_dim):
        raise NotImplementedError

    @abstractclassmethod
    def __len__(self):
        raise NotImplementedError

    @abstractclassmethod
    def sample(self):
        raise NotImplementedError

    @abstractclassmethod
    def merge(self, local_buffer, length, start=0):
        raise NotImplementedError
