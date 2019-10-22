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
            self.no = worker_no                             # use 0 worker to evaluate the model
            self.weight_update_freq = weight_update_freq    # update weights 
            buffer_args['type'] = 'local'
            buffer_args['local_capacity'] = 1 if worker_no == 0 else env_args['max_episode_steps'] * weight_update_freq

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
            if self.no == 0:
                scores = deque(maxlen=self.weight_update_freq)
                epslens = deque(maxlen=self.weight_update_freq)
                best_score_mean = 0
            episode_i = 0
            t = 0
            while True:
                episode_i += 1
                state = self.env.reset()
                for _ in range(self.max_path_length):
                    t += 1
                    if self.no == 0:
                        action = self.act(state, deterministic=True)
                    else:
                        action = self.act(state)
                    next_state, reward, done, _ = self.env.step(action, self.max_action_repetition)
                    
                    if self.no != 0:
                        self.buffer.add_data(state, action, reward, done)

                    state = next_state

                    if done:
                        break

                if episode_i % self.weight_update_freq == 0:
                    if self.no == 0:
                        # for worker_0, log score
                        score = self.env.get_score()
                        epslen = self.env.get_epslen()
                        scores.append(score)
                        epslens.append(epslen)
                        score_mean = np.mean(scores)
                        stats = dict(score=score, score_mean=score_mean, score_std=np.std(scores),
                                    epslen=epslen, epslen_mean=np.mean(epslens), epslen_std=np.std(scores),
                                    worker_no=self.no, global_step=episode_i)
                                    
                        learner.record_stats.remote(stats)
                        if score_mean > best_score_mean:
                            best_score_mean = score_mean
                            self.save()
                    else:
                        # for other workers, send data to learner
                        last_state = np.zeros_like(state) if done else next_state
                        self.buffer.add_last_state(last_state)
                        self.buffer['priority'] = self.compute_priorities()
                        # push samples to the central buffer after each episode
                        learner.merge_buffer.remote(dict(self.buffer), self.buffer.idx)
                        self.buffer.reset()
                    
                    # pull weights from learner
                    weights = ray.get(learner.get_weights.remote())
                    self.variables.set_flat(weights)

        def print_construction_complete(self):
            pwc(f'Worker {self.no} has been constructed.', 'cyan')
            
    return Worker.remote(*args, **kwargs)
