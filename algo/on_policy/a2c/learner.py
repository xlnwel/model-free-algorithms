import numpy as np
import tensorflow as tf
import ray

from utility.utils import normalize, pwc
from algo.on_policy.ppo.agent import Agent


@ray.remote(num_cpus=1, num_gpus=0.1)
class Learner(Agent):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 save=True,
                 log=False,
                 log_tensorboard=False,
                 log_params=False,
                 log_stats=False,
                 device=None):
        super().__init__(name, 
                         args, 
                         env_args, 
                         sess_config=sess_config,
                         save=save, 
                         log=log,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_stats=log_stats,
                         device=device)
        del self.buffer
    
    def apply_gradients(self, timestep, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {g_var: g for g_var, g in zip(self.ac.grads, grads)}

        # do not log_tensorboard, use record_stats if required
        learn_step, _, _ = self.sess.run([self.ac.opt_step, self.ac.policy_optop, self.ac.v_optop], 
                                         feed_dict=feed_dict)

        if hasattr(self, 'saver') and learn_step % 100 == 0:
            self.save()

        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()

    def record_stats(self, score_mean, score_std, epslen_mean, entropy, approx_kl, clip_frac):
        # a wraper since ray does not support (*args)
        super().record_stats(score_mean=score_mean, score_std=score_std,
                             epslen_mean=epslen_mean, entropy=entropy,
                             approx_kl=approx_kl, clip_frac=clip_frac)

    def print_construction_complete(self):
        pwc('Learner has been constructed.', color='cyan')
