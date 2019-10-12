import numpy as np


# Reference code:
# https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        batch_mean = np.squeeze(batch_mean)
        batch_var = np.squeeze(batch_var)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # print(f'self:\tmean:{self.mean:.2f}\tvar:{self.var:.2f}\tcount:{self.count}')
        # print(f'batch:\tmean:{batch_mean:.2f}\tvar:{batch_var:.2f}\tcount:{batch_count}')
        if self.count == self.epsilon:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / total_count
            # no minus one here. To honor the original implementation, I leave it as-is
            m_a = self.var * (self.count)
            m_b = batch_var * (batch_count)
            # print(f'ma:{m_a:.2f}\tmb:{m_b:.2f}\tdelta:{delta:.2f}')
            # print(delta**2, self.count * batch_count, self.count + batch_count)
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            assert not np.isinf(M2) and not np.isnan(M2), f'M2: {M2}'
            new_var = M2 / (self.count + batch_count - 1)

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count

    def normalize(self, x):
        assert not np.isinf(np.std(x)), f'{np.min(x)}\t{np.max(x)}'
        # print(f'before normalization:\tmean:{np.mean(x)}\tstd:{np.std(x)}')
        x = (x - self.mean) / (self.var + self.epsilon)
        # print(f'after normalization:\tmean:{np.mean(x)}\tstd:{np.std(x)}')
        return x
