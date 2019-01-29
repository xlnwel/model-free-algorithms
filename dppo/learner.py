import numpy as np
import ray

from agent import Agent
from utils.np_math import norm

@ray.remote
class Learner(Agent):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=True,
                 log_tensorboard=False,
                 log_params=False):

        self.timestep = 0
        super().__init__(name, args, env_args, sess_config=sess_config,
                         reuse=reuse, save=save, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)
    
    def apply_gradients(self, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {grad_and_var[0]: grad for grad_and_var, grad in zip(self.grads_and_vars, grads)}
       
        _, summary = self.sess.run([self.opt_op, self.graph_summary], feed_dict=feed_dict)

        self.timestep += 1
        if self.timestep % 10 == 0:
            self.writer.add_summary(summary, self.timestep)
            self.save()

        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()
     