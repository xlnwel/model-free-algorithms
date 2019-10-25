from __future__ import absolute_import, division, print_function, unicode_literals  # provide backward compatibility

import time
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from ray.experimental.tf_utils import TensorFlowVariables

from utility.logger import Logger
from utility.utils import pwc
from utility.debug_tools import assert_colorize
from basic_model.model import Model
from env.gym_env import create_env
from algo.off_policy.apex.buffer import LocalBuffer
from algo.off_policy.replay.uniform_replay import UniformReplay
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
                 save=False, 
                 log=False,
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        # hyperparameters
        self.gamma = args.setdefault('gamma', .99)
        self.update_step = 0
        self.max_action_repetitions = args.setdefault('max_action_repetitions', 1)

        # environment info
        env_args['gamma'] = self.gamma
        env_args['seed'] += 100
        self.eval_env = create_env(env_args)
        env_args['seed'] -= 100
        env_args['log_video'] = False
        self.train_env = create_env(env_args)
        self.state_space = self.train_env.state_space
        self.action_dim = self.train_env.action_dim

        # replay buffer hyperparameters
        buffer_args['n_steps'] = args['n_steps']
        buffer_args['gamma'] = args['gamma']
        buffer_args['batch_size'] = args['batch_size']

        if buffer_args['type'] == 'proportional':
            self.buffer = ProportionalPrioritizedReplay(buffer_args, self.state_space, self.action_dim)
        elif buffer_args['type'] == 'uniform':
            self.buffer = UniformReplay(buffer_args, self.state_space, self.action_dim)
        elif buffer_args['type'] == 'local':
            self.buffer = LocalBuffer(buffer_args, self.state_space, self.action_dim)
        else:
            raise NotImplementedError
        
        # arguments for prioritized replay
        self.prio_alpha = float(buffer_args['alpha'])
        self.prio_epsilon = float(buffer_args['epsilon'])

        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log=log,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

        self._initialize_target_net()

        with self.graph.as_default():
            self.variables = TensorFlowVariables(self.loss, self.sess)
        
    @property
    def max_path_length(self):
        return self.train_env.max_episode_steps
    
    @property
    def good_to_learn(self):
        return self.buffer.good_to_learn

    def add_data(self, state, action_repr, reward, done):
        self.buffer.add(state, action_repr, reward, done)

    def merge_buffer(self, buffer, length):
        self.buffer.merge(buffer, length)

    def act(self, state, deterministic=False):
        raise NotImplementedError

    def run_trajectory(self, fn=None, render=False, random_action=False, deterministic_action=False):
        raise NotImplementedError

    def learn(self, t=None):
        if self.schedule_lr:
            assert t is not None
            feed_dict = self._get_feeddict(t)
        else:
            feed_dict = None
        if self.log_tensorboard:
            priority, saved_mem_idxs, _, summary = self.sess.run([self.priority, 
                                                                  self.data['saved_mem_idxs'], 
                                                                  self.opt_op, 
                                                                  self.graph_summary],
                                                                  feed_dict=feed_dict)
            if self.update_step % 1000 == 0:
                self.writer.add_summary(summary, self.update_step)
        else:
            priority, saved_mem_idxs, _ = self.sess.run([self.priority, 
                                                         self.data['saved_mem_idxs'], 
                                                         self.opt_op],
                                                         feed_dict=feed_dict)

        if self.update_step % 10000 == 0 and hasattr(self, 'saver'):
            self.save()
        # update the target networks
        self._update_target_net()

        self.update_step += 1
        self.buffer.update_priorities(priority, saved_mem_idxs)
    
    def rl_log(self, kwargs):
        assert isinstance(kwargs, dict)
        assert 'Timing' in kwargs
        assert 'Episodes' in kwargs or 'Steps' in kwargs
        assert 'Score' in kwargs
        assert 'ScoreMean' in kwargs
        assert 'ScoreStd' in kwargs

        log_info = {
            'ModelName': f'{self.args["algorithm"]}-{self.model_name}'
        }
        log_info.update(kwargs)

        [self.log_tabular(k, v) for k, v in log_info.items()]
        self.dump_tabular()

    """ Implementation """
    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError

    def _prepare_data(self, buffer):
        with tf.name_scope('data'):
            exp_type = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
            sample_types = (tf.float32, tf.int32, exp_type)
            action_shape = (None, self.action_dim)

            sample_shapes =((None), (None), (
                (None, *self.state_space),
                action_shape,
                (None, 1),
                (None, *self.state_space),
                (None, 1),
                (None, 1)
            ))
            ds = tf.data.Dataset.from_generator(buffer, sample_types, sample_shapes)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            iterator = ds.make_one_shot_iterator()
            samples = iterator.get_next(name='samples')
        
        # prepare data
        IS_ratio, saved_mem_idxs, (state, action, reward, next_state, done, steps) = samples

        data = {}
        data['IS_ratio'] = IS_ratio[:, None]                 # Importance sampling ratio for PER
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

    def _get_feeddict(self, t):
        raise NotImplementedError
