from __future__ import absolute_import, division, print_function, unicode_literals  # provide backward compatibility

import time
import numpy as np
import tensorflow as tf
import ray

from basic_model.model import Model
from gym_env.env import GymEnvironment
from actor_critic import Actor, Critic, DoubleCritic
from utility.losses import huber_loss
from replay.local_buffer import LocalBuffer
from replay.proportional_replay import ProportionalPrioritizedReplay


class Agent(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 reuse=None, 
                 save=True, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_score=False, 
                 device=None):
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99 
        self.tau = args['tau'] if 'tau' in args else 1e-3

        # options for DDPG improvements
        options = args['options']
        self.n_steps = options['n_steps']

        self.critic_loss_type = args['critic']['loss_type']
        self.extra_critic_updates = args['critic']['extra_updates']

        # environment info
        self.env = GymEnvironment(env_args['name'])
        self.max_path_length = (env_args['max_episode_steps'] if 'max_episode_steps' in env_args 
                                 else self.env.max_episode_steps)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        
        # replay buffer
        if buffer_args['type'] == 'proportional':
            self.buffer = ProportionalPrioritizedReplay(buffer_args, self.state_dim, self.action_dim)
        elif buffer_args['type'] == 'local':
            self.buffer = LocalBuffer(buffer_args['store_episodes'] * self.max_path_length)

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

        with self._graph.as_default():
            self.variables = ray.experimental.TensorFlowVariables([self.actor_loss, self.critic_loss], self.sess)

    @property
    def main_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    @property
    def target_variables(self):
        return self._target_actor.trainable_variables + self._target_critic.trainable_variables

    def act(self, state):
        state = state.reshape((-1, self.state_dim))
        action = self.sess.run(self.actor.action, feed_dict={self.actor.state: state})

        return np.squeeze(action)

    def add_data(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def background_learning(self):
        while not self.buffer.good_to_learn:
            time.sleep(1)
        print('Start Learning...')
        while True:
            self.learn()

    def learn(self):
        # update the main networks
        for _ in range(self.extra_critic_updates):
            priority, saved_exp_ids, _ = self.sess.run([self.priority,
                                                        self.data['saved_exp_ids'],
                                                        self.critic_opt_op])
            self.buffer.update_priorities(priority, saved_exp_ids)

        if self._log_tensorboard:
            priority, saved_exp_ids, global_step, _, _, summary = self.sess.run([self.priority, 
                                                                                self.data['saved_exp_ids'],
                                                                                self.global_step, 
                                                                                self.actor_opt_op, 
                                                                                self.critic_opt_op, 
                                                                                self.graph_summary])
            if global_step % 100 == 0:
                self.writer.add_summary(summary, global_step)
                self.save()
        else:
            priority, saved_exp_ids, _, _ = self.sess.run([self.priority, self.data['saved_exp_ids'], self.actor_opt_op, self.critic_opt_op])

        self.buffer.update_priorities(priority, saved_exp_ids)

        # update the target networks
        self.sess.run(self.update_target_op)
    
    """ Implementation """
    def _build_graph(self):
        if 'gpu' in self._device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)
            
        self.actor, self.critic, self._target_actor, self._target_critic = self._create_main_target_actor_critic()

        self.priority, self.actor_loss, self.critic_loss = self._loss()
    
        self.actor_opt_op, self.global_step = self.actor._optimization_op(self.actor_loss, global_step=True)
        self.critic_opt_op, _ = self.critic._optimization_op(self.critic_loss)

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_loss()

    def _prepare_data(self, buffer):
        with tf.name_scope('data'):
            sample_types = (tf.float32, tf.int32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            sample_shapes =((None), (None), (
                (None, self.state_dim),
                (None, self.action_dim),
                (None, 1),
                (None, self.state_dim),
                (None, 1),
                (None, 1)
            ))
            ds = tf.data.Dataset.from_generator(buffer, sample_types, sample_shapes)
            ds = ds.prefetch(1)
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

    def _create_main_target_actor_critic(self):
        # main actor-critic
        actor, critic = self._create_actor_critic(is_target=False)
        # target actor-critic
        target_actor, target_critic = self._create_actor_critic(is_target=True)

        return actor, critic, target_actor, target_critic
        
    def _create_actor_critic(self, is_target):
        log_tensorboard = False if is_target else self._log_tensorboard
        log_params = False if is_target else self._log_params

        scope_name = 'target' if is_target else 'main'
        state = self.data['next_state'] if is_target else self.data['state']
        scope_prefix = self.name + '/' + scope_name
        
        with tf.variable_scope(scope_name, reuse=self._reuse):
            actor = Actor('actor', 
                          self._args['actor'], 
                          self._graph,
                          state, 
                          self.action_dim, 
                          reuse=self._reuse, 
                          scope_prefix=scope_prefix, 
                          log_tensorboard=log_tensorboard, 
                          log_params=log_params)

            critic = DoubleCritic('critic', 
                                 self._args['critic'],
                                 self._graph,
                                 state,
                                 self.data['action'], 
                                 actor.action,
                                 self.action_dim,
                                 reuse=self._reuse, 
                                 scope_prefix=scope_prefix, 
                                 log_tensorboard=log_tensorboard,
                                 log_params=log_params)
        
        return actor, critic

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                actor_loss = -tf.reduce_mean(self.critic.Q1_with_actor * self.data['IS_ratio'])

            with tf.name_scope('critic_loss'):
                priority, critic_loss = self._double_critic_loss()

        return priority, actor_loss, critic_loss

    def _double_critic_loss(self):
        target_Q = self._n_step_target(self._target_critic.Q_with_actor)
        
        TD_error1 = tf.abs(target_Q - self.critic.Q1, name='TD_error1')
        TD_error2 = tf.abs(target_Q - self.critic.Q2, name='TD_error2')
        with tf.name_scope(name='priority'):
            priority = self._compute_priorities((TD_error1 + TD_error2) / 2.)

        loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
        TD_squared = loss_func(TD_error1) + loss_func(TD_error2)

        critic_loss = tf.reduce_mean(self.data['IS_ratio'] * TD_squared)

        return priority, critic_loss

    def _compute_priorities(self, priorities):
        priorities += self.prio_epsilon
        priorities **= self.prio_alpha
        
        return priorities

    def _n_step_target(self, nth_Q):
        n_step_target = tf.add(self.data['reward'], self.gamma**self.data['steps']
                                                    * (1 - self.data['done'])
                                                    * nth_Q, name='n_step_target')

        return tf.stop_gradient(n_step_target)

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

    def _initialize_target_net(self):
        self.sess.run(self.init_target_op)

    def _log_loss(self):
        if self._log_tensorboard:
            with tf.name_scope('priority'):
                tf.summary.histogram('priority_', self.priority)
                tf.summary.scalar('priority_', tf.reduce_mean(self.priority))

            with tf.name_scope('IS_ratio'):
                tf.summary.histogram('IS_ratio_', self.data['IS_ratio'])
                tf.summary.scalar('IS_ratio_', tf.reduce_mean(self.data['IS_ratio']))

            with tf.variable_scope('loss', reuse=self._reuse):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('critic_loss_', self.critic_loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.critic.Q_with_actor))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.critic.Q_with_actor))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.critic.Q_with_actor))
            