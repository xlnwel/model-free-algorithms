import os
import time
from collections import deque
import numpy as np
import threading
import tensorflow as tf
import ray

from utility import tf_utils
from utility.utils import pwc


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
            weights = self.variables.get_flat()

            return weights

        def merge_buffer(self, local_buffer, length):
            self.buffer.merge(local_buffer, length)

        def background_learning(self):
            while not self.buffer.good_to_learn:
                time.sleep(1)
            pwc('Start Learning...', color='cyan')
            
            i = 0
            scores = deque(maxlen=100)
            epslens = deque(maxlen=100)
            while True:
                i += 1
                self.learn()
                if i % 1000 == 0:
                    pwc('Start Evaluation...', color='cyan')
                    for _ in range(100):
                        score, epslen = self.run_trajectory(lambda: None, test=True)
                    self.rl_log(Timing='Eval',
                                Steps=i,
                                Score=score,
                                ScoreMean=np.mean(scores),
                                ScoreStd=np.std(scores),
                                EpsLenMean=np.mean(epslens),
                                EpsLenStd=np.std(epslens))

        def record_stats(self, kwargs):
            assert isinstance(kwargs, dict)
            super()._record_stats_impl(kwargs)

        def print_construction_complete(self):
            pwc('Learner has been constructed.', color='cyan')

    return Learner.remote(*args, **kwargs)
