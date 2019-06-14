from __future__ import absolute_import, division, print_function, unicode_literals  # provide backward compatibility

import time
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from ray.experimental.tf_utils import TensorFlowVariables

from utility.logger import Logger
from utility.debug_tools import assert_colorize
from basic_model.model import Model
from env.gym_env import GymEnv, GymEnvVec
from algo.off_policy.apex.buffer import LocalBuffer
from algo.off_policy.replay.proportional_replay import ProportionalPrioritizedReplay


class OffPolicyOperation(Model, ABC):
    """ Abstract base class for off-policy algorithms.
    Generally speaking, inherited class only need to define _build_graph
    and leave all interface as it-is.
    """
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=True, 
                 log_tensorboard=True, 
                 log_params=False, 
                 log_stats=True, 
                 device=None):
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99
        self.prefetches = args['prefetches'] if 'prefetches' in args else 0
        self.update_step = 0

        # environment info
        self.env = (GymEnvVec(env_args) if 'n_envs' in env_args and env_args['n_envs'] > 1
                    else GymEnv(env_args))
        self.state_space = self.env.state_space
        self.action_dim = self.env.action_dim
        
        # replay buffer
        buffer_args['n_steps'] = args['n_steps']
        buffer_args['gamma'] = args['gamma']
        buffer_args['batch_size'] = args['batch_size']

        # if action is discrete, then it has only 1 dimension
        action_dim = 1 if self.env.is_action_discrete else self.action_dim
        if buffer_args['type'] == 'proportional':
            self.buffer = ProportionalPrioritizedReplay(buffer_args, self.env.state_space, action_dim)
        elif buffer_args['type'] == 'local':
            self.buffer = LocalBuffer(buffer_args, self.env.state_space, action_dim)

        # arguments for prioritized replay
        self.prio_alpha = float(buffer_args['alpha'])
        self.prio_epsilon = float(buffer_args['epsilon'])

        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

        self._initialize_target_net()

        with self.graph.as_default():
            self.variables = TensorFlowVariables(self.loss, self.sess)
        
    @property
    def max_path_length(self):
        return self.env.max_episode_steps
    
    def act(self, state, return_q=False):
        state = state.reshape((-1, *self.state_space))
        if return_q:
            action, q = self.sess.run([self.action, self.critic.Q_with_action], 
                                        feed_dict={self.data['state']: state})
            return np.squeeze(action), q
        else:
            action = self.sess.run(self.action, feed_dict={self.data['state']: state})
        
        return np.squeeze(action)

    def add_data(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def background_learning(self):
        from utility.debug_tools import timeit
        while not self.buffer.good_to_learn:
            time.sleep(1)
        print('Start Learning...')
        
        i = 0
        lt = []
        while True:
            i += 1
            duration, _ = timeit(self.learn)
            lt.append(duration)
            if i % 1000 == 0:
                print(f'{self.model_name}:\tTakes {np.sum(lt):3.2f}s to learn 1000 times')
                i = 0
                lt = []

    def learn(self):
        if self.log_tensorboard:
            priority, saved_mem_idxs, _, summary = self.sess.run([self.priority, 
                                                                self.data['saved_mem_idxs'], 
                                                                self.opt_op, 
                                                                self.graph_summary])
            if self.update_step % 100 == 0:
                self.writer.add_summary(summary, self.update_step)
                self.save()
        else:
            priority, saved_mem_idxs, _ = self.sess.run([self.priority, 
                                                        self.data['saved_mem_idxs'], 
                                                        self.opt_op])

        # update the target networks
        self._update_target_net()

        self.update_step += 1
        self.buffer.update_priorities(priority, saved_mem_idxs)
    
    """ Implementation """
    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError

    def _prepare_data(self, buffer):
        with tf.name_scope('data'):
            if self.env.is_action_discrete:
                exp_type = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
            else:
                exp_type = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
            sample_types = (tf.float32, tf.int32, exp_type)
            action_shape =(None, ) if self.env.is_action_discrete else (None, self.action_dim)
            sample_shapes =((None), (None), (
                (None, *self.state_space),
                action_shape,
                (None, 1),
                (None, *self.state_space),
                (None, 1),
                (None, 1)
            ))
            ds = tf.data.Dataset.from_generator(buffer, sample_types, sample_shapes)
            if self.prefetches > 0:
                ds = ds.prefetch(self.prefetches)
            iterator = ds.make_one_shot_iterator()
            samples = iterator.get_next(name='samples')
        
        # prepare data
        IS_ratio, saved_mem_idxs, (state, action, reward, next_state, done, steps) = samples

        data = {}
        data['IS_ratio'] = IS_ratio                 # Importance sampling ratio for PER
        # saved indexes used to index the experience in the buffer when updating priorities
        data['saved_mem_idxs'] = saved_mem_idxs     
        data['state'] = state
        data['action'] = action
        data['reward'] = reward
        data['next_state'] = next_state
        data['done'] = done
        data['steps'] = steps

        return data

    def _compute_priority(self, priority):
        with tf.name_scope('priority'):
            priority += self.prio_epsilon
            priority **= self.prio_alpha
        
        return priority

    def _initialize_target_net(self):
        raise NotImplementedError
    
    def _update_target_net(self):
        raise NotImplementedError

