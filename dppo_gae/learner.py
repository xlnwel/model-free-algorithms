import numpy as np
import ray

from ppo_gae import PPOGAE

@ray.remote
class Learner(PPOGAE):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=True,
                 log_tensorboard=False):

        super().__init__(name, args, env_args, sess_config=sess_config,
                         reuse=reuse, save=save, log_tensorboard=log_tensorboard)
    
    def apply_gradients(self, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {grad_and_var[0]: grad for grad_and_var, grad in zip(self.grads_and_vars, grads)}
       
        self.sess.run(self.opt_op, feed_dict=feed_dict)
        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()
        