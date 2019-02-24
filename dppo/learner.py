import numpy as np
import tensorflow as tf
import ray

from utility.utils import normalize
from dppo.agent import Agent


@ray.remote#(num_cpus=0.5, num_gpus=0.04)
class Learner(Agent):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=True,
                 log_tensorboard=False,
                 log_params=False,
                 log_score=False,
                 device=None):

        super().__init__(name, args, env_args, sess_config=sess_config,
                         reuse=reuse, save=save, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_score=log_score,
                         device=device)
    
    def apply_gradients(self, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {grad_and_var[0]: grad for grad_and_var, grad in zip(self.grads_and_vars, grads)}
        
        learn_step, _, summary = self.sess.run([self.global_step, self.opt_op, self.graph_summary], feed_dict=feed_dict)

        if self._log_tensorboard and learn_step % 10 == 0:
            self.writer.add_summary(summary, learn_step)
            self.save()

        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()

    def log_score(self, score, avg_score, save=True):
        super().log_score(score, avg_score)
        if save:
            self.save()
