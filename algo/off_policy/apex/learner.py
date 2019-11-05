import os
import time
from collections import deque
import numpy as np
import threading
import tensorflow as tf
import ray

from utility import tf_utils
from utility.display import pwc


def get_learner(BaseClass, *args, **kwargs):
    @ray.remote(num_gpus=0.3, num_cpus=2)
    class Learner(BaseClass):
        """ Interface """
        def __init__(self, 
                    name, 
                    args, 
                    env_args,
                    buffer_args,
                    sess_config=None, 
                    save=False, 
                    log=False,
                    log_tensorboard=False, 
                    log_params=False, 
                    log_stats=False, 
                    device=None):
            env_args['n_envs'] = 1
            super().__init__(name, 
                            args, 
                            env_args,
                            buffer_args,
                            sess_config=sess_config,
                            save=save,
                            log=log,
                            log_tensorboard=log_tensorboard,
                            log_params=log_params,
                            log_stats=log_stats,
                            device=device)
            
            self.learning_thread = threading.Thread(target=self.background_learning, daemon=True)
            self.learning_thread.start()
            
        def get_weights(self):
            return self.variables.get_flat()

        def set_weights(self, weights):
            pwc('Learner: pull weights from the evaluator', 'blue')
            self.variables.set_flat(weights)

        def merge_buffer(self, local_buffer, length):
            self.buffer.merge(local_buffer, length)

        def background_learning(self):
            while not self.buffer.good_to_learn:
                time.sleep(1)
            pwc('Start Learning...', 'blue')
            
            t = 0
            while True:
                t += 1
                self.learn(t)

        def record_stats(self, kwargs):
            assert isinstance(kwargs, dict)
            super()._record_stats_impl(kwargs)

        def print_construction_complete(self):
            pwc('Learner has been constructed.', 'cyan')

    return Learner.remote(*args, **kwargs)
