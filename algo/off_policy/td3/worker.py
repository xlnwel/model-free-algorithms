from time import time
from collections import deque
import numpy as np
import ray

from utility.utils import pwc
from replay.utils import reset_buffer, add_buffer
from algo.off_policy.td3.agent import Agent


@ray.remote(num_cpus=.5)
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
                         save=save,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_score=log_score,
                         device=device)

        self.max_episodes = max_episodes
        self.local_buffer_capacity = max_episodes * self.max_path_length
        self.lb_idx = 0

        basic_shape = (self.local_buffer_capacity, 1)
        self.buffer.update({
            'state': np.zeros((self.local_buffer_capacity, *self.state_space)),
            'action': np.zeros((self.local_buffer_capacity, self.action_space)),
            'reward': np.zeros(basic_shape),
            'next_state': np.zeros((self.local_buffer_capacity, *self.state_space)),
            'done': np.zeros(basic_shape),
            'steps': np.zeros(basic_shape),
            'priority': np.zeros(basic_shape),
        })
        
        pwc('Worker {} has been constructed.'.format(self.no), 'cyan')

    def sample_data(self, learner):
        # I intend not to synchronize the worker's weights at the beginning for initial exploration 
        score_deque = deque(maxlen=100)
        eps_len_deque = deque(maxlen=100)
        episode_i = 0
        
        while True:
            state = self.env.reset()
            
            for _ in range(self.max_path_length):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                add_buffer(self.buffer, self.lb_idx, state, action, reward, 
                            next_state, done, self.n_steps, self.gamma)

                state = next_state

                self.lb_idx += 1

                if done:
                    break
            
            score = self.env.get_episode_score()
            eps_len = self.env.get_episode_length()
            episode_i += 1
            score_deque.append(score)
            eps_len_deque.append(eps_len)
            learner.log_stats.remote(score=score, avg_score=np.mean(score_deque), 
                                     eps_len=eps_len, avg_eps_len=np.mean(eps_len_deque), 
                                     worker_no=self.no)
            
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
