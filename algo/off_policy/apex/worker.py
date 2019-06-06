import os
from time import time
from collections import deque
import numpy as np
import ray

from utility.utils import pwc


def get_worker(BaseClass, *args, **kwargs):

    @ray.remote(num_cpus=1)
    class Worker(BaseClass):
        """ Interface """
        def __init__(self, 
                    name, 
                    worker_no,
                    args, 
                    env_args,
                    buffer_args,
                    max_episodes,
                    sess_config=None, 
                    save=False, 
                    log_tensorboard=False, 
                    log_params=False,
                    log_score=False,
                    device=None):
            self.no = worker_no

            super().__init__(name, 
                            args, 
                            env_args,
                            buffer_args,
                            sess_config=sess_config,
                            save=save,
                            log_tensorboard=log_tensorboard,
                            log_params=log_params,
                            log_score=log_score,
                            device=device)

            self.max_episodes = max_episodes
            pwc('Worker {} has been constructed.'.format(self.no), 'cyan')

        def sample_data(self, learner):
            # I intend not to synchronize the worker's weights at the beginning for initial exploration 
            score_deque = deque(maxlen=100)
            eps_len_deque = deque(maxlen=100)
            episode_i = 0
            
            while True:
                state = self.env.reset()

                for _ in range(self.max_path_length):
                    action = self.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    self.buffer.add(state, action, reward, next_state, done)

                    if self.buffer.is_full:
                        priority = self.sess.run(self.priority)
                        self.buffer['priority'] = priority

                        learner.merge_buffer.remote(dict(self.buffer), self.buffer.capacity)
                        self.buffer.reset()

                    state = next_state

                    if done:
                        break

                score = self.env.get_score()
                eps_len = self.env.get_length()
                episode_i += 1
                score_deque.append(score)
                eps_len_deque.append(eps_len)
                stats = dict(score=score, avg_score=np.mean(score_deque), 
                            eps_len=eps_len, avg_eps_len=np.mean(eps_len_deque), 
                            worker_no=self.no)
                            
                learner.log_stats.remote(stats)
                
                # pull weights from learner
                if episode_i >= self.max_episodes:
                    weights = ray.get(learner.get_weights.remote())
                    self.variables.set_flat(weights)
                    episode_i = 0

    return Worker.remote(*args, **kwargs)
