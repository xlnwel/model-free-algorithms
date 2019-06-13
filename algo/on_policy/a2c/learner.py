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
                 log_tensorboard=True,
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
    
    def apply_gradients(self, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {g_var: g for g_var, g in zip(self.ac.grads, grads)}
        
        if self.graph_summary is not None:
            learn_step, _, summary = self.sess.run([self.ac.opt_step, self.ac.opt_op, self.graph_summary], feed_dict=feed_dict)
        else:
            learn_step, _ = self.sess.run([self.ac.opt_step, self.ac.opt_op], feed_dict=feed_dict)

        if self.log_tensorboard and learn_step % 100 == 0:
            if self.graph_summary:
                self.writer.add_summary(summary, learn_step)
            self.save()

        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()

    def record_stats(self, score, avg_score, eps_len, avg_eps_len, worker_no):
        super().record_stats(score=score, avg_score=avg_score, 
                          eps_len=eps_len, avg_eps_len=avg_eps_len, 
                          worker_no=worker_no)
