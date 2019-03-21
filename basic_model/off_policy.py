from __future__ import absolute_import, division, print_function, unicode_literals  # provide backward compatibility

import time
import numpy as np
import tensorflow as tf
import ray

from utility.logger import Logger
from basic_model.model import Model
from env.gym_env import GymEnvironment
from replay.local_buffer import LocalBuffer
from replay.proportional_replay import ProportionalPrioritizedReplay


class OffPolicy(Model):
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
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99 
        self.tau = args['tau'] if 'tau' in args else 1e-3
        self.policy_delay = args['policy_delay'] if 'policy_delay' in args else 1
        self.prefetches = args['prefetches']
        self.update_step = 0

        # environment info
        self.env = GymEnvironment(env_args['name'], args['model_name'], atari=False)
        self.max_path_length = (env_args['max_episode_steps'] if 'max_episode_steps' in env_args 
                                 else self.env.max_episode_steps)
        self.state_space = self.env.state_space
        self.action_space = self.env.action_space
        
        # replay buffer
        if buffer_args['type'] == 'proportional':
            self.buffer = ProportionalPrioritizedReplay(buffer_args, self.state_space, self.action_space)
        elif buffer_args['type'] == 'local':
            self.buffer = LocalBuffer(buffer_args['max_episodes'] * self.max_path_length)

        # arguments for prioritized replay
        self.prio_alpha = float(buffer_args['alpha'])
        self.prio_epsilon = float(buffer_args['epsilon'])

        super().__init__(name, args, 
                         sess_config=sess_config, 
                         reuse=reuse,
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_score=log_score, 
                         device=device)

        self._initialize_target_net()

        with self.graph.as_default():
            self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
    
    def act(self, state, return_q=False):
        state = state.reshape((-1, *self.state_space))
        # if return_q
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
                print(f'{self.model_name}:\tTakes {np.sum(lt):3.2f} to learn 1000 times')
                i = 0
                lt = []

    def learn(self):
        # update the main networks
        if self.update_step % self.policy_delay != 0:
            priority, saved_exp_ids, _ = self.sess.run([self.priority,
                                                        self.data['saved_exp_ids'],
                                                        self.critic_opt_op])
        else:
            if self.log_tensorboard:
                priority, saved_exp_ids, global_step, _, summary = self.sess.run([self.priority, 
                                                                                self.data['saved_exp_ids'],
                                                                                self.global_step, 
                                                                                self.opt_op, 
                                                                                self.graph_summary])
                if global_step % 100 == 0:
                    self.writer.add_summary(summary, global_step)
                    self.save()
            else:
                priority, saved_exp_ids, _ = self.sess.run([self.priority, 
                                                            self.data['saved_exp_ids'], 
                                                            self.opt_op])

            # update the target networks
            self._update_target_net()

        self.update_step = (self.update_step + 1) % self.policy_delay
        self.buffer.update_priorities(priority, saved_exp_ids)
    
    """ Implementation """
    def _prepare_data(self, buffer):
        with tf.name_scope('data'):
            sample_types = (tf.float32, tf.int32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            sample_shapes =((None), (None), (
                (None, *self.state_space),
                (None, self.action_space),
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
        IS_ratio, saved_exp_ids, (state, action, reward, next_state, done, steps) = samples

        data = {}
        data['IS_ratio'] = IS_ratio
        data['saved_exp_ids'] = saved_exp_ids
        data['state'] = state
        data['action'] = action
        data['reward'] = reward
        data['next_state'] = next_state
        data['done'] = done
        data['steps'] = steps

        return data

    def _n_step_target(self, nth_value):
        n_step_target = tf.stop_gradient(self.data['reward'] 
                                        + self.gamma**self.data['steps']
                                        * (1 - self.data['done'])
                                        * nth_value, name='n_step_target')

        return n_step_target

    def _compute_priority(self, priority):
        with tf.name_scope('priority'):
            priority += self.prio_epsilon
            priority **= self.prio_alpha
        
        return priority

    def _initialize_target_net(self):
        raise NotImplementedError
    
    def _update_target_net(self):
        raise NotImplementedError

