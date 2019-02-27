from collections import deque

from td3_rainbow.agent import Agent


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

        self.local_buffer = self._reset_local_buffer()
        self.local_buffer_capacity = 1e3
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

                self._add_buffer(self.local_buffer, state, action, reward, next_state, done)

                if self.local_buffer['counter'] > self.local_buffer_capacity + self.n_steps:
                    self.buffer.merge.remote(self.local_buffer)
                    self._reset_local_buffer()
                    break
            
            if i > self.weight_update_steps:
                weights_id = self.learner.get_weights.remote()
                self._set_weights(weights_id)
                i = 0


    def _set_weights(self, weights):
        self.variables.set_flat(weights)

    def _reset_local_buffer(self):
        return {
            'counter': 0,
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': [],
            'steps': []
        }

    def _add_buffer(self, buffer, state, action, reward, next_state, done):
        buffer['counter'] += 1
        buffer['state'].append(state)
        buffer['action'].append(action)
        buffer['reward'].append(reward)
        buffer['next_state'].append(next_state),
        buffer['done'].append(done)
        buffer['steps'].append(1)
        for i in range(1, self.n_steps):
            if buffer['done'][-idx] == True:
                break
            idx = i + 1
            buffer['steps'][-idx] += 1
            buffer['reward'][-idx] += self.gamma**i * reward
