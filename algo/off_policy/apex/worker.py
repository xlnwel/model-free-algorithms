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
                    weight_update_freq,
                    sess_config=None, 
                    save=False, 
                    device=None):
            self.no = worker_no
            self.weight_update_freq = weight_update_freq    # update weights 
            buffer_args['type'] = 'local'
            buffer_args['local_capacity'] = env_args['max_episode_steps']

            super().__init__(name, 
                            args, 
                            env_args,
                            buffer_args,
                            sess_config=sess_config,
                            save=save,
                            device=device)

        def compute_priorities(self):
            return self.sess.run(self.priority)

        def sample_data(self, learner):
            # I intend not to synchronize the worker's weights at the beginning for initial diversity 
            score_deque = deque(maxlen=100)
            epslen_deque = deque(maxlen=100)
            episode_i = 0
            t = 0
            
            while True:
                state = self.env.reset()

                for _ in range(self.max_path_length):
                    t += 1
                    action = self.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    self.buffer.add(state, action, reward, done)

                    state = next_state

                    if done:
                        break

                last_state = np.zeros_like(state) if done else next_state
                self.buffer.add_last_state(last_state)
                self.buffer['priority'] = self.compute_priorities()
                # push samples to the central buffer after each episode
                learner.merge_buffer.remote(dict(self.buffer), self.buffer.idx)
                self.buffer.reset()

                score = self.env.get_score()
                epslen = self.env.get_length()
                episode_i += 1
                score_deque.append(score)
                epslen_deque.append(epslen)
                stats = dict(score=score, score_mean=np.mean(score_deque), score_std=np.std(score_deque),
                             epslen=epslen, epslen_mean=np.mean(epslen_deque), epslen_std=np.std(score_deque),
                             worker_no=self.no)
                            
                learner.record_stats.remote(stats)
                
                # pull weights from learner
                if episode_i >= self.weight_update_freq:
                    weights = ray.get(learner.get_weights.remote())
                    self.variables.set_flat(weights)
                    episode_i = 0

        def print_construction_complete(self):
            pwc(f'Worker {self.no} has been constructed.', 'cyan')
            
    return Worker.remote(*args, **kwargs)
