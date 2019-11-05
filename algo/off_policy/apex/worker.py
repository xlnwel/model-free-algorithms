import os
from time import time
from collections import deque
import numpy as np
import ray

from utility.display import pwc
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
            state, action, reward, next_state, done, steps = self.buffer.sample()
            return self.sess.run(self.priority, feed_dict={
                self.data['state']: state,
                self.data['action']: action,
                self.data['reward']: reward,
                self.data['next_state']:next_state,
                self.data['done']: done,
                self.data['steps']: steps
            })

        def sample_data(self, learner, evaluator):
            def collect_fn(state, action, reward, done):
                self.buffer.add_data(state, action, reward, done)

            def pull_weights_from_learner():
                # pull weights from learner
                weights = ray.get(learner.get_weights.remote())
                self.variables.set_flat(weights)

            to_record = self.no == 0
            scores = deque(maxlen=self.weight_update_freq)
            epslens = deque(maxlen=self.weight_update_freq)
            best_score_mean = -50
            episode_i = 0
            step = 0
            while True:
                episode_i += 1
                fn = None if to_record else collect_fn
                score, epslen = self.run_trajectory(fn=fn, evaluation=to_record)
                step += epslen
                scores.append(score)
                epslens.append(epslen)

                if episode_i % self.weight_update_freq == 0:
                    score_mean = np.mean(scores)
                    if to_record:
                        # record stats
                        stats = dict(
                            Timing='Eval',
                            WorkerNo=self.no,
                            Steps=episode_i,
                            ScoreMean=score_mean, 
                            ScoreStd=np.std(scores),
                            ScoreMax=np.max(score), 
                            EpslenMean=np.mean(epslens), 
                            EpslenStd=np.std(epslens), 
                        )
                        tf_stats = dict(worker_no=f'worker_{self.no}')
                        tf_stats.update(stats)

                        learner.record_stats.remote(tf_stats)
                        
                        learner.rl_log.remote(stats)

                    if score_mean > min(250, best_score_mean):
                        best_score_mean = score_mean
                        pwc(f'Worker {self.no}: Best score updated to {best_score_mean:2f}', 'blue')
                        evaluator.evaluate_model.remote(self.variables.get_flat(), score_mean)

                    # send data to learner
                    if self.buffer.idx == self.buffer.capacity:
                        last_state = np.zeros_like(self.buffer['state'][0])
                        self.buffer.add_last_state(last_state)
                        self.buffer['priority'][:self.buffer.idx] = self.compute_priorities()
                        # push samples to the central buffer after each episode
                        learner.merge_buffer.remote(dict(self.buffer), self.buffer.idx)
                        self.buffer.reset()

                    # pull weights from learner
                    weights = ray.get(learner.get_weights.remote())
                    self.variables.set_flat(weights)

        def print_construction_complete(self):
            pwc(f'Worker {self.no} has been constructed.', 'cyan')

    return Worker.remote(*args, **kwargs)
