import numpy as np
import tensorflow as tf
import ray

from utility.utils import normalize
from algo.on_policy.ppo.agent import Agent


@ray.remote(num_cpus=1, num_gpus=0.1)
class Learner(Agent):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 save=True,
                 log_tensorboard=False,
                 log_params=False,
                 log_stats=True,
                 device=None):
        super().__init__(name, 
                         args, 
                         env_args, 
                         sess_config=sess_config,
                         save=save, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_stats=log_stats,
                         device=device)
    
    def apply_gradients(self, timestep, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {g_var: g for g_var, g in zip(self.ac.grads, grads)}
        if self.ac.args['schedule_lr']:
            feed_dict[self.ac.learning_rate] = self.ac.lr_scheduler.value(timestep)
            
        if self.log_tensorboard:
            learn_step, _, summary = self.sess.run([self.ac.opt_step, self.ac.opt_op, self.graph_summary], feed_dict=feed_dict)
        else:
            learn_step, _ = self.sess.run([self.ac.opt_step, self.ac.opt_op], feed_dict=feed_dict)

        if self.log_tensorboard and learn_step % 100 == 0:
            self.writer.add_summary(summary, learn_step)
            if hasattr(self, 'saver'):
                self.save()

        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()

    def record_stats(self, score_mean, score_std, epslen_mean, entropy, approx_kl, clip_frac):
        # a wraper since ray does not support (*args)
        super().record_stats(score_mean=score_mean, score_std=score_std,
                             epslen_mean=epslen_mean, entropy=entropy,
                             approx_kl=approx_kl, clip_frac=clip_frac)
