import numpy as np
import tensorflow as tf
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

        # rl-specific log configuration, not in self._build_graph to avoid being included in self.graph_summary
        if self._log_tensorboard:
            self.score, self.avg_score, self.score_counter, self.score_log_op = self._setup_score_logs()
        
            # initialize score_counter
            self.sess.run(tf.variables_initializer([self.score_counter]))

        # reset saver to include score counter
        self._saver = self._setup_saver(save)
    
    def apply_gradients(self, *grads):
        grads = np.mean(grads, axis=0)
        
        feed_dict = {grad_and_var[0]: grad for grad_and_var, grad in zip(self.grads_and_vars, grads)}
       
        _, summary = self.sess.run([self.opt_op, self.graph_summary], feed_dict=feed_dict)

        self.timestep += 1
        if self._log_tensorboard and self.timestep % 10 == 0:
            self.writer.add_summary(summary, self.timestep)
            self.save()

        return self.get_weights()

    def get_weights(self):
        return self.variables.get_flat()

    def log_score(self, score, avg_score):
        if self._log_tensorboard:
            feed_dict = {
                self.score: score,
                self.avg_score: avg_score
            }

            score_count, summary = self.sess.run([self.score_counter, self.score_log_op], feed_dict=feed_dict)
            self.writer.add_summary(summary, score_count)
            
    """ Implementation """ 
    def _setup_score_logs(self):
        with self._graph.as_default():
            with tf.variable_scope('scores', reuse=self._reuse):
                score = tf.placeholder(tf.float32, shape=None, name='score')
                avg_score = tf.placeholder(tf.float32, shape=None, name='average_score')

                score_counter = tf.get_variable('score_counter', shape=[], initializer=tf.constant_initializer(), trainable=False)
                step_op = tf.assign(score_counter, score_counter + 1, name='update_score_counter')
                
                score_log = tf.summary.scalar('score_', score)
                avg_score_log = tf.summary.scalar('average_score_', avg_score)

                with tf.control_dependencies([step_op]):
                    score_log_op = tf.summary.merge([score_log, avg_score_log], name='score_log_op')

        return score, avg_score, score_counter, score_log_op