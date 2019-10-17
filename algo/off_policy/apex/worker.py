import os
from time import time
from collections import deque
import numpy as np
import ray

from utility.utils import pwc
from utility.schedule import PiecewiseSchedule


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
            buffer_args['local_capacity'] = env_args['max_episode_steps'] * weight_update_freq

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
            train_scores = deque(maxlen=100)
            train_epslens = deque(maxlen=100)
            eval_scores = deque(maxlen=100)
            eval_epslens = deque(maxlen=100)
            episode_i = 0
            t = 0
            if self.no != 0: # code for BipedalWalkerHardcore-v2
                # schedule the episode length
                episode_len_schedular = PiecewiseSchedule([(0, 1000), (1000, 1500), (2000, 2000)])
            while True:
                state = self.env.reset()
                episode_len = self.max_path_length if self.no == 0 else int(episode_len_schedular.value(episode_i))
                for _ in range(episode_len):
                    t += 1
                    action = self.act(state, deterministic=self.no == 0)
                    next_state, reward, done, _ = self.env.step(action, self.max_action_repetition)
                    
                    self.buffer.add_data(state, action, reward, done)

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
                epslen = self.env.get_epslen()
                episode_i += 1
                train_scores.append(score)
                train_epslens.append(epslen)
                stats = dict(score=score, score_mean=np.mean(train_scores), score_std=np.std(train_scores),
                             epslen=epslen, epslen_mean=np.mean(train_epslens), epslen_std=np.std(train_scores),
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
