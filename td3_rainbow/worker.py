from collections import deque
import numpy as np
import ray

from replay.utils import reset_buffer, add_buffer
from td3_rainbow.agent import Agent


class LocalBuffer(dict):
    def __init__(self, length, **kwarg):
        self.fake_ratio = np.zeros(length)
        self.fake_ids = np.zeros(length, dtype=np.int32)

    def __call__(self):
        while True:
            yield self.fake_ratio, self.fake_ids, (self['state'], self['action'], self['reward'], 
                                self['next_state'], self['done'], self['steps'])


@ray.remote(num_cpus=1)
class Worker(Agent):
    """ Interface """
    def __init__(self, 
                 name, 
                 worker_no,
                 args, 
                 env_args,
                 buffer_args,
                 learner,
                 weight_update_interval,
                 sess_config=None, 
                 reuse=None, 
                 save=False, 
                 log_tensorboard=False, 
                 log_params=False,
                 log_score=False,
                 device=None):
        self.no = worker_no
        self.local_buffer_capacity = int(1e3)
        self.buffer = LocalBuffer(self.local_buffer_capacity)
        self.lb_idx = 0
        self.learner = learner
        self.weight_update_interval = weight_update_interval

        super().__init__(name, args, env_args,
                         buffer_args=buffer_args,
                         sess_config=sess_config,
                         reuse=reuse, 
                         save=save,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_score=log_score,
                         device=device)

        def reset():
            reset_buffer(self.buffer, self.local_buffer_capacity + self.n_steps, 
                                                self._state_dim, self._action_dim, True)
        self._reset_buffer = lambda: reset_buffer(self.buffer, self.local_buffer_capacity + self.n_steps, 
                                                self._state_dim, self._action_dim, True)
        self._reset_buffer()

    def sample_data(self):
        i = 0
        avg_score = deque(maxlen=100)
        while True:
            state = self.env.reset()
            score = 0
            for _ in range(self._max_path_length):
                i += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward

                add_buffer(self.buffer, self.lb_idx, state, action, reward, 
                            next_state, done, self.n_steps, self.gamma)
                self.lb_idx += 1

                if self.buffer['counter'] >= self.local_buffer_capacity + self.n_steps:
                    priority = self.sess.run(self.priority)
                    self.buffer['priority'] = priority

                    # buffer_id = ray.put(self.buffer)
                    self.learner.merge_buffer.remote(dict(self.buffer), self.local_buffer_capacity)

                    self._reset_buffer()
                    self.lb_idx = 0
                    break
                    
                if done:
                    break

            avg_score.append(score)

            self.learner.log_score.remote(self.no, score, np.mean(avg_score))
            
            if i > self.weight_update_interval:
                weights = ray.get(self.learner.get_weights.remote())
                self.variables.set_flat(weights)
                i = 0

        return self.no