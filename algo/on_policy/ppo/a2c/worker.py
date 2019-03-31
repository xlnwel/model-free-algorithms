import numpy as np
import ray

from utility.utils import pwc
from algo.on_policy.ppo.buffer import PPOBuffer
from algo.on_policy.ppo.agent import Agent


@ray.remote(num_cpus=1, num_gpus=0.09)
class Worker(Agent):
    """ Interface """
    def __init__(self,
                 name,
                 worker_no,
                 args,
                 env_args,
                 sess_config=None,
                 save=False,
                 device=None):
        super().__init__(name, 
                         args, 
                         env_args, 
                         sess_config=sess_config, 
                         save=save, 
                         device=device)
        self.no = worker_no
        self.entropy_coef = self.args['ac']['entropy_coef']
        self.n_minibatches = args['n_minibatches']
        self.minibach_size = self.seq_len // self.n_minibatches
        self.minibatch_idx = 0
        if self.use_lstm:
            pwc('lstm is used', 'red')
            self.last_lstm_state = None

        self.buffer = PPOBuffer(args['n_envs'], self.seq_len, self.n_minibatches, self.minibach_size
                                self.env_vec.state_space, self.env_vec.action_dim, shuffle=args['shuffle'])
        pwc('Worker {} has been constructed.'.format(self.no), 'cyan')

    @ray.method(num_return_vals=2)
    def compute_gradients(self, weights):
        self._set_weights(weights)

        fetches = [[grad_and_var[0] for grad_and_var in self.ac.grads_and_vars], 
                   [self.ac.ppo_loss, self.ac.entropy, 
                    self.ac.V_loss, self.ac.loss, 
                    self.ac.approx_kl, self.ac.clipfrac]]
        feed_dict = {}
        if self.use_lstm:
            fetches.append(self.ac.final_state)
            feed_dict[self.ac.initial_state] = self.last_lstm_state

        # normalize advantages (& returns)
        feed_dict.update({
            self.env_phs['state']: self.buffer.get_flat_batch('state', self.minibatch_idx),
            self.ac.action: self.buffer.get_flat_batch('action', self.minibatch_idx),
            self.env_phs['return']: self.buffer.get_flat_batch('return', self.minibatch_idx),
            self.env_phs['advantage']: self.buffer.get_flat_batch('advantage', self.minibatch_idx),
            self.env_phs['old_logpi']: self.buffer.get_flat_batch('old_logpi', self.minibatch_idx),
            self.env_phs['entropy_coef']: self.entropy_coef
        })

        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_lstm:
            grads, loss_info, self.last_lstm_state = results
        else:
            grads, loss_info = results
        self.minibatch_idx = self.minibatch_idx + 1 if self.minibatch_idx + 1 < self.n_minibatches else 0
        
        return grads, loss_info

    def sample_trajectories(self, weights):
        # function content
        self._set_weights(weights)
        env_stats = self._sample_data()
        self.buffer.compute_ret_adv(self.args['advantage_type'], self.gamma, self.gae_discount)

        return env_stats

    def act(self, state):
        fetches = [self.ac.action, self.ac.V, self.ac.logpi]
        feed_dict = {self.env_phs['state']: state}
        if self.use_lstm:
            fetches.append(self.ac.final_state)
            feed_dict.update({self.ac.initial_state: self.last_lstm_state})
        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_lstm:
            action, value, logpi, self.last_lstm_state = results
        else:
            action, value, logpi = results

        return action, np.squeeze(value), np.squeeze(logpi)

    def get_advantages(self):
        return self.buffer['advantage']

    def normalize_advantages(self, mean, std):
        self.buffer['advantage'] = (self.buffer['advantage'] - mean) / (std + 1e8)

    """ Implementation """
    def _set_weights(self, weights):
        self.variables.set_flat(weights)

    def _sample_data(self):
        self.buffer.reset()
        state = self.env_vec.reset()

        if self.use_lstm:
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: state})
        
        for _ in range(self.seq_len):
            action, value, logpi = self.act(state)
            next_state, reward, done, _ = self.env_vec.step(action)

            self.buffer.add(state, action, reward, value, logpi, 1-np.array(done))

            state = next_state
        
        # add one more ad hoc value so that we can take values[1:] as next state values
        value = np.squeeze(self.sess.run(self.ac.V, feed_dict={self.env_phs['state']: state}))
        self.buffer['value'][:, -1] = value
        
        return self.env_vec.get_episode_score(), self.env_vec.get_episode_length()
