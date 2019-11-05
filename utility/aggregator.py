class Aggregator:
    """Allows accumulating values and computing their mean."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0

    def average(self):
        return self.sum / self.count if self.count else 0.

    def add(self, v):
        self.sum += v
        self.count += 1