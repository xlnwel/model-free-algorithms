from time import time
from collections import deque
import numpy as np
import ray

from replay.utils import reset_buffer, add_buffer
from td3.agent import Agent
from utility import tf_utils


@ray.remote(num_cpus=1)
class Worker(Agent):
    """ Interface """
    def __init__(self, 
                 name, 
                 worker_no,
                 args, 
                 env_args,
                 buffer_args,
                 max_episodes,
                 sess_config=None, 
                 reuse=None, 
                 save=False, 
                 log_tensorboard=False, 
                 log_params=False,
                 log_score=False,
                 device=None):
        self.no = worker_no
        buffer_args['max_episodes'] = max_episodes

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

        self.max_episodes = max_episodes
        self.local_buffer_capacity = max_episodes * self.max_path_length
        self.lb_idx = 0

        self.buffer.update({
            'state': np.zeros((self.local_buffer_capacity, self.state_dim)),
            'action': np.zeros((self.local_buffer_capacity, self.action_dim)),
            'reward': np.zeros((self.local_buffer_capacity, 1)),
            'next_state': np.zeros((self.local_buffer_capacity, self.state_dim)),
            'done': np.zeros((self.local_buffer_capacity, 1)),
            'steps': np.zeros((self.local_buffer_capacity, 1)),
            'priority': np.zeros((self.local_buffer_capacity, 1))
        })
        
        print('Worker {} has been constructed.'.format(self.no))

    def sample_data(self, learner):
        # I intend not to synchronize the worker's weights at the beginning for initial exploration 
        score_deque = deque(maxlen=100)
        episode_i = 0
        
        while True:
            state = self.env.reset()
            score = 0
            start = time()
            
            for _ in range(self.max_path_length):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                add_buffer(self.buffer, self.lb_idx, state, action, reward, 
                            next_state, done, self.n_steps, self.gamma)

                score += reward
                state = next_state

                self.lb_idx += 1

                if done:
                    break
            
            episode_i += 1
            score_deque.append(score)
            learner.log_score.remote(self.no, score, np.mean(score_deque))
            
            if episode_i >= self.max_episodes:
                priority = self.sess.run(self.priority)
                self.buffer['priority'] = priority

                # buffer_id = ray.put(self.buffer)
                learner.merge_buffer.remote(dict(self.buffer), self.lb_idx)

                reset_buffer(self.buffer)
                self.lb_idx = 0
                episode_i = 0

                weights = ray.get(learner.get_weights.remote(self.no))
                self.variables.set_flat(weights)
