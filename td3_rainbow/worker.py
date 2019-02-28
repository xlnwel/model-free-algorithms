from collections import deque

from td3_rainbow.agent import Agent
from td3_rainbow.replay.utils import init_buffer, add_buffer


class Worker(Agent):
    def __init__(self, 
                 name, 
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
        self.local_buffer_capacity = int(1e3)
        self.local_buffer = init_buffer(self.local_buffer_capacity + self.n_steps, self.state_dim, self.action_dim)
        self.lb_idx = 0
        self.learner = learner
        self.weight_update_steps = weight_update_steps
        # TODO: move to agent
        with self._graph.as_default():
            self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)

    def sample_data(self):
        i = 0
        for _ range(3000):
            state = self.env.reset()
            for _ in range(self._max_path_length):
                i += 1
                action = self.act(ob)
                next_state, reward, done, _ = self.env.step(action)

                add_buffer(self.local_buffer, self.lb_idx, state, action, reward, 
                            next_state, done, self.n_steps, self.gamma)

                if self.local_buffer['counter'] >= self.local_buffer_capacity + self.n_steps:
                    self.buffer.merge.remote(self.local_buffer, self.local_buffer_capacity)
                    self.local_buffer = init_buffer(self.local_buffer_capacity + self.n_steps)
                    break
            
            if i > self.weight_update_steps:
                weights_id = self.learner.get_weights.remote()
                self._set_weights(weights_id)
                i = 0


    def _set_weights(self, weights):
        self.variables.set_flat(weights)

    def _add_buffer(self, buffer, state, action, reward, next_state, done):
        buffer['counter'] += 1
        buffer['state'].append(state)
        buffer['action'].append(action)
        buffer['reward'].append(reward)
        buffer['next_state'].append(next_state),
        buffer['done'].append(done)
        buffer['steps'].append(1)
        for i in range(1, self.n_steps):
            k = i + 1
            if buffer['done'][-k] == True or k > buffer['counter']:
                break
            buffer['reward'][-k] += self.gamma**i * reward
            buffer['next_state'][-k] = next_state
            buffer['done'][-k] = done
            buffer['steps'][-k] += 1
