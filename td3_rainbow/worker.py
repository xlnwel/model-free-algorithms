from collections import deque
import ray

from td3_rainbow.agent import Agent
from replay.utils import init_buffer, add_buffer


@ray.remote
class Worker(Agent):
    def __init__(self, 
                 name, 
                 worker_no,
                 args, 
                 env_args,
                 buffer,
                 learner,
                 weight_update_steps,
                 sess_config=None, 
                 reuse=None, 
                 save=True, 
                 log_tensorboard=False, 
                 log_params=False,
                 log_score=False,
                 device=None):
        super().__init__(name, args, env_args,
                         buffer, sess_config,
                         reuse, save,
                         log_tensorboard,
                         log_params,
                         log_score,
                         device) 
        self.no = worker_no
        self.local_buffer_capacity = int(1e3)
        self._init_buffer = lambda: init_buffer(self.local_buffer_capacity + self.n_steps, self._state_dim, self._action_dim, True)
        self.local_buffer = self._init_buffer()
        self.lb_idx = 0
        self.learner = learner
        self.weight_update_steps = weight_update_steps

    def sample_data(self):
        i = 0
        while True:
            state = self.env.reset()
            for _ in range(self._max_path_length):
                i += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                add_buffer(self.local_buffer, self.lb_idx, state, action, reward, 
                            next_state, done, self.n_steps, self.gamma)

                if self.local_buffer['counter'] >= self.local_buffer_capacity + self.n_steps:
                    priority = self.sess.run(self.priority)
                    self.local_buffer['priority']= priority

                    buffer_id = ray.put(self.local_buffer)
                    self.buffer.merge.remote(buffer_id, self.local_buffer_capacity)
                    self.local_buffer = self._init_buffer()
                    self.lb_idx = 0
                    break
            
            if i > self.weight_update_steps:
                weights_id = self.learner.get_weights.remote()
                self._set_weights(weights_id)
                i = 0


    def _set_weights(self, weights):
        self.variables.set_flat(weights)
