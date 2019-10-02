import numpy as np
import ray

from utility.utils import pwc
from algo.on_policy.ppo.buffer import PPOBuffer
from algo.on_policy.ppo.agent import Agent


@ray.remote(num_cpus=1, num_gpus=0.1)
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

        self.no = worker_no
        self.minibatch_idx = 0

        super().__init__(name, 
                         args, 
                         env_args, 
                         sess_config=sess_config, 
                         save=save, 
                         device=device)

    @ray.method(num_return_vals=2)
    def compute_gradients(self, weights):
        self._set_weights(weights)

        # construct fetches
        fetches = [self.ac.grads, 
                   [self.ac.ppo_loss, self.ac.entropy, 
                    self.ac.approx_kl, self.ac.clipfrac,
                    self.ac.V_loss]]
        
        if self.use_lstm:
            fetches.append(self.ac.final_state)

        # construct feed_dict
        feed_dict = self._get_feeddict()

        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_lstm:
            grads, loss_info, self.last_lstm_state = results
        else:
            grads, loss_info = results
        self.minibatch_idx = (self.minibatch_idx + 1) % self.n_minibatches
        
        return grads, loss_info

    def sample_trajectories(self, weights):
        # function content
        self._set_weights(weights)

        env_stats = self._sample_data()
        
        return env_stats

    def get_advantages(self):
        return self.buffer['advantage']

    def normalize_advantages(self, mean, std):
        self.buffer['advantage'] = (self.buffer['advantage'] - mean) / (std + 1e8)

    def print_construction_complete(self):
        pwc(f'Worker {self.no} has been constructed.', 'cyan')
            

    """ Implementation """
    def _set_weights(self, weights):
        self.variables.set_flat(weights)

