import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from utility import tf_utils
from basic_model.model import Model
from actor_critic import Actor, Critic, DoubleCritic
from utility.debug_tools import timeit
from utility.losses import huber_loss
from gym_env.env import GymEnvironment

# different replay buffers
from td3_rainbow.replay.uniform_replay import UniformReplay
from td3_rainbow.replay.rank_based_replay import RankBasedPrioritizedReplay
from td3_rainbow.replay.proportional_replay import ProportionalPrioritizedReplay

class Agent(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args,
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
        self.batch_size = args['batch_size'] if 'batch_size' in args else 128

        # options for DDPG improvements
        options = args['options']
        self._buffer_type = options['buffer_type']
        self._double = options['double']
        self.n_steps = options['n_steps']

        self._critic_loss = args['critic']['loss_type']
        self._critic_update_times = args['critic']['update_times']

        # environment info
        self.env = GymEnvironment(env_args['name'])
        self._max_path_length = (env_args['max_episode_steps'] if 'max_episode_steps' in env_args 
                                 else self.env.max_episode_steps)
        self._observation_dim = self.env.observation_dim
        self._action_dim = self.env.action_dim
        super().__init__(name, args, sess_config=sess_config, 
                         reuse=reuse, save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params,
                         log_score=log_score,
                         device=device)

        # replay buffer
        self.buffer = self._select_buffer(self._buffer_type, self.batch_size)

        self._initialize_target_net()

    @property
    def main_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    @property
    def _target_variables(self):
        return self._target_actor.trainable_variables + self._target_critic.trainable_variables

    def act(self, observation):
        observation = observation.reshape((-1, self.env.observation_dim))
        action = self.sess.run(self.actor.action, feed_dict={self.actor.observations_ph: observation})

        return np.squeeze(action)

    def learn(self, observation, action, reward, next_observation, done):
        self.buffer.add(observation, action, reward, next_observation, done)

        if self.trainable and self.buffer.good_to_learn:
            self._learn()

    """ Implementation """
    def _build_graph(self, **kwargs):
        # environment info
        self.env_phs = self._setup_env_placeholders(self._observation_dim, self._action_dim)
        
        self.actor, self.critic, self._target_actor, self._target_critic = self._create_main_target_actor_critic()

        self.priorities, self.actor_loss, self.critic_loss = self._loss()
    
        self.actor_optimizer, self.global_step = self.actor._adam_optimizer(global_step=True)
        self.critic_optimizer, critic_step = self.critic._adam_optimizer()

        self.actor_grads_vars = self.actor._compute_gradients(self.actor_loss, self.actor_optimizer, self.actor.trainable_variables)
        self.critic_grads_vars = self.critic._compute_gradients(self.critic_loss, self.critic_optimizer, self.critic.trainable_variables)

        self.actor_opt_op = self.actor._apply_gradients(self.actor_optimizer, self.actor_grads_vars, self.global_step)
        self.critic_opt_op = self.critic._apply_gradients(self.critic_optimizer, self.critic_grads_vars, critic_step)
        
        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_loss()

    def _setup_env_placeholders(self, observation_dim, action_dim):
        env_phs = {}
        with tf.name_scope('placeholder'):
            env_phs['observation'] = tf.placeholder(tf.float32, shape=(None, observation_dim), name='observation')
            env_phs['action'] = tf.placeholder(tf.float32, shape=(None, action_dim), name='action')
            env_phs['reward'] = tf.placeholder(tf.float32, shape=(None, self.n_steps, 1), name='reward')
            env_phs['next_observation'] = tf.placeholder(tf.float32, shape=(None, observation_dim), name='next_observation')
            env_phs['done'] = tf.placeholder(tf.uint8, shape=(None, 1), name='done')
            env_phs['steps'] = tf.placeholder(tf.uint8, shape=(None, 1), name='steps')
            if self._buffer_type != 'uniform':
                env_phs['IS_ratio'] = tf.placeholder(tf.float32, shape=(self.batch_size), name='importance_sampling_ratio')
        
        return env_phs

    def _create_main_target_actor_critic(self):
        # main actor-critic
        actor, critic = self._create_actor_critic(is_target=False, is_double=self._double)
        # target actor-critic
        target_actor, target_critic = self._create_actor_critic(is_target=True, is_double=self._double)

        return actor, critic, target_actor, target_critic
        
    def _create_actor_critic(self, is_target, is_double):
        scope_name = 'target' if is_target else 'main'
        scope_prefix = self.name + '/' + scope_name
        with tf.variable_scope(scope_name, reuse=self._reuse):
            actor = Actor('actor', 
                          self._args['actor'], 
                          self._graph, 
                          self.env_phs['observation'], 
                          self._action_dim,
                          reuse=self._reuse,
                          is_target=is_target,
                          scope_prefix=scope_prefix)

            critic = (DoubleCritic if is_double else Critic)('critic', 
                                                             self._args['critic'], 
                                                             self._graph,
                                                             self.env_phs['observation'], 
                                                             self.env_phs['action'], 
                                                             actor.action,
                                                             self._action_dim,
                                                             reuse=self._reuse,
                                                             is_target=is_target, 
                                                             scope_prefix=scope_prefix)
        
        return actor, critic

    """ Losses """
    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                Q_with_actor = self.critic.Q_with_actor
                actor_loss = tf.negative(tf.reduce_mean(Q_with_actor), name='actor_loss')

            with tf.name_scope('critic_loss'):
                priorities, critic_loss = self._double_critic_loss() if self._double else self._plain_critic_loss()
            
        return priorities, actor_loss, critic_loss

    def _double_critic_loss(self):
        target_Q = self._n_step_target(self._target_critic.Q_with_actor)
        
        TD_error1 = tf.abs(target_Q - self.critic.Q1, name='TD_error1')
        TD_error2 = tf.abs(target_Q - self.critic.Q2, name='TD_error2')
        priorities = tf.divide(TD_error1 + TD_error2, 2., name='priorities')

        loss_func = huber_loss if self._critic_loss == 'huber' else tf.square
        TD_squared = loss_func(TD_error1) + loss_func(TD_error2)
        
        critic_loss = self._average_critic_loss(TD_squared)

        return priorities, critic_loss
        
    def _plain_critic_loss(self):
        target_Q = self._n_step_target(self._target_critic.Q_with_actor)
        
        TD_error = tf.abs(target_Q - self.critic.Q, name='TD_error')
        priorities = tf.identity(TD_error, name='priorities')

        loss_func = huber_loss if self._critic_loss == 'huber' else tf.square
        TD_squared = loss_func(TD_error)

        critic_loss = self._average_critic_loss(TD_squared)
        
        return priorities, critic_loss

    def _average_critic_loss(self, loss):
        weighted_loss = loss if self._buffer_type == 'uniform' else self.env_phs['IS_ratio'] * loss
        
        critic_loss = tf.reduce_mean(weighted_loss, name='critic_loss')

        return critic_loss

    def _n_step_target(self, n_step_value):
        rewards_sum = tf.reduce_sum(self.env_phs['reward'], axis=1)
        n_step_gamma = self.gamma**tf.cast(self.env_phs['steps'], tf.float32)
        n_step_target = tf.add(rewards_sum, n_step_gamma
                                            * tf.cast(1 - self.env_phs['done'], tf.float32)
                                            * n_step_value, name='target_Q')
        
        return tf.stop_gradient(n_step_target)

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self._target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

    def _select_buffer(self, buffer_type, batch_size):
        buffer_args = self._args['buffer']
        
        if buffer_type == 'uniform':
            buffer = UniformReplay(buffer_args, batch_size, n_steps=self.n_steps, gamma=self.gamma)
        elif buffer_type == 'rank':
            buffer = RankBasedPrioritizedReplay(buffer_args, batch_size, n_steps=self.n_steps, gamma=self.gamma)
        elif buffer_type == 'proportional':
            buffer = ProportionalPrioritizedReplay(buffer_args, batch_size, n_steps=self.n_steps, gamma=self.gamma)
        else:
            raise ValueError('Invalid buffer type.')

        return buffer

    def _learn(self):
        def env_feed_dict(buffer):
            IS_ratios, (observation, actions, rewards, next_observations, dones, steps) = buffer.sample()
            if self._buffer_type == 'uniform':
                feed_dict = {}
            else:
                feed_dict = {self.env_phs['IS_ratio']: IS_ratios}

            feed_dict.update({
                self.env_phs['observation']: observation,
                self.env_phs['action']: actions,
                self.env_phs['reward']: rewards,
                self.env_phs['next_observation']: next_observations,
                self.env_phs['done']: dones,
                self.env_phs['steps']: steps
            })

            return feed_dict

        # update critic a few times first
        for _ in range(self._critic_update_times-1):
            feed_dict = env_feed_dict(self.buffer)
            self.sess.run([self.critic_opt_op], feed_dict=feed_dict)

        feed_dict = env_feed_dict(self.buffer)
        # update the main networks
        if self._log_tensorboard:
            priorities, learn_steps, _, _, summary = self.sess.run([self.priorities, 
                                                                    self.global_step, 
                                                                    self.critic_opt_op, 
                                                                    self.actor_opt_op, 
                                                                    self.graph_summary], 
                                                                   feed_dict=feed_dict)
            if learn_steps % 100 == 0:
                self.writer.add_summary(summary, learn_steps)
                self.save()
        else:
            priorities, _, _ = self.sess.run([self.priorities, self.critic_opt_op, self.actor_opt_op], 
                                          feed_dict=feed_dict)

        if self._buffer_type != 'uniform':
            self.buffer.update_priorities(priorities)

        # update the target networks
        self.sess.run(self.update_target_op)

    def _initialize_target_net(self):
        self.sess.run(self.init_target_op)

    def _log_loss(self):
        if self._log_tensorboard:
            if self._buffer_type != 'uniform':
                with tf.name_scope('priority'):
                    tf.summary.histogram('priorities_', self.priorities)
                    tf.summary.scalar('priority_', tf.reduce_mean(self.priorities))

                with tf.name_scope('IS_ratio'):
                    tf.summary.histogram('IS_ratios_', self.env_phs['IS_ratio'])
                    tf.summary.scalar('IS_ratio_', tf.reduce_mean(self.env_phs['IS_ratio']))

            with tf.variable_scope('loss', reuse=self._reuse):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('critic_loss_', self.critic_loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.critic.Q_with_actor))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.critic.Q_with_actor))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.critic.Q_with_actor))
            