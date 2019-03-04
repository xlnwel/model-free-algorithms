from time import sleep
import numpy as np
import threading
import tensorflow as tf
import ray


from td3_rainbow.agent import Agent

@ray.remote(num_gpus=1, num_cpus=1)
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
                 log_params=False,
                 log_score=True,
                 device=None):
        super().__init__(name, args, env_args,
                         buffer_args, sess_config,
                         reuse, save,
                         log_tensorboard,
                         log_params,
                         log_score,
                         device)
        self.learning_thread = threading.Thread(target=self._background_learning, args=())
        self.learning_thread.start()
        
    def get_weights(self):
        return self.variables.get_flat()
    
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

    """ Implementation """
    def _background_learning(self):
        while not self.buffer.good_to_learn:
            sleep(1)
        i = 0
        print('Start Learning')
        while True:
            i += 1
            print('\rLearning step: {}'.format(i))
            self._learn()
