from time import sleep
import numpy as np
import threading
import tensorflow as tf
import ray


from td3.agent import Agent
from utility import tf_utils

@ray.remote(num_gpus=.1, num_cpus=2)
class Learner(Agent):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args,
                 buffer_args,
                 sess_config=None, 
                 reuse=None, 
                 save=True, 
                 log_tensorboard=True, 
                 log_params=True,
                 log_score=True,
                 device=None):
        super().__init__(name, 
                         args, 
                         env_args,
                         buffer_args,
                         sess_config=sess_config,
                         reuse=reuse, 
                         save=save,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_score=log_score,
                         device=device)
        self.net_locker = threading.Lock()
        self.learning_thread = threading.Thread(target=self.background_learning)
        self.learning_thread.start()
        
        print('Learner has been constructed.')
        
    def get_weights(self, no):
        self.net_locker.acquire()
        weights = self.variables.get_flat()
        self.net_locker.release()

        return weights
    
    def log_score(self, worker_no, score, avg_score):
        feed_dict = {
            self.scores[worker_no]: score,
            self.avg_scores[worker_no]: avg_score
        }

        score_count, summary = self.sess.run([self.score_counters[worker_no], self.score_log_ops[worker_no]], 
                                            feed_dict=feed_dict)
        self.writer.add_summary(summary, score_count)

    def merge_buffer(self, local_buffer, length):
        self.buffer.merge(local_buffer, length)
        
    def learn(self):
        self.net_locker.acquire()
        super().learn()
        self.net_locker.release()
