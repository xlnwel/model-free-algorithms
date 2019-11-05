from time import strftime, gmtime, time
from collections import defaultdict
import tensorflow as tf

from utility.aggregator import Aggregator


def timeit(func, *args, name=None, to_print=False, **kwargs):
	start_time = gmtime()
	start = time()
	result = func(*args, **kwargs)
	end = time()
	end_time = gmtime()

	if to_print:
		print(f'{name if name else func.__name__}: '
              f'Start "{strftime("%d %b %H:%M:%S", start_time)}"', 
              f'End "{strftime("%d %b %H:%M:%S", end_time)}"' 
              f'Duration "{end - start:.2f}s"')

	return end - start, result

class Timer:
    def __init__(self, summary_name):
        self.summary_name = summary_name

    def __enter__(self):
        self.start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        duration = time() - self.start
        print(f'{self.summary_name} duration "{duration:.2f}s"')


class TFTimer:
    aggregators = defaultdict(Aggregator)

    def __init__(self, summary_name, period):
        self.summary_name = summary_name
        self.period = period

    def __enter__(self):
        self.start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        duration = time() - self.start
        aggregator = self.aggregators[self.summary_name]
        aggregator.add(duration)
        if aggregator.count >= self.period:
            tf.summary.scalar(self.summary_name, aggregator.average())
            aggregator.reset()

class LoggerTimer:
    def __init__(self, logger, summary_name):
        self.logger = logger
        self.summary_name = summary_name

    def __enter__(self):
        self.start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        duration = time() - self.start
        self.logger.store(**{self.summary_name: duration})
