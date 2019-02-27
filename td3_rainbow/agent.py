import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utility import tf_utils
from basic_model.model import Model
from actor_critic import Actor, Critic, DoubleCritic
from utility.debug_tools import timeit
from utility.losses import huber_loss

# different replay buffers
from replay.uniform_replay import UniformReplay
from replay.rank_based_replay import RankBasedPrioritizedReplay
from replay.proportional_replay import ProportionalPrioritizedReplay

class Agent(Model):
    """ Interface """
    def __init__(self, name, args, env_args, buffer, sess_config=None, reuse=None, log_tensorboard=True, save=True, log_params=False, log_score=True, device=None):
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99 
        self.tau = args['tau'] if 'tau' in args else 1e-3
        self.batch_size = args['batch_size'] if 'batch_size' in args else 128

        # options for DDPG improvements
        options = args['options']
        self._buffer_type = options['buffer_type']
        self._double = options['double_Q']
        self._distributional = options['distributional']
        self.n_steps = options['n_steps']

        self._critic_loss = args['critic']['loss_type']

        # replay buffer
        self.buffer = buffer

        super().__init__(name, args, sess_config=sess_config, reuse=reuse,
                         log_tensorboard=log_tensorboard, save=save, 
                         log_params=log_params, log_score=log_score, device=device)

        self._initialize_target_net()

    @property
    def main_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    @property
    def _target_variables(self):
        return self._target_actor.trainable_variables + self._target_critic.trainable_variables

    def act(self, state):
        state = state.reshape((-1, self.state_size))
        action = self.sess.run(self.actor.action, feed_dict={self.actor.state: state})

        return np.squeeze(action)

    def learn(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        if self.trainable and self.buffer.good_to_learn:
            self._learn()
    
    """ Implementation """
    def _build_graph(self, **kwargs):
        # environment info
        self.state_size = self._args['state_size']
        self.action_size = self._args['action_size']

        self.env_info = self._setup_env_placeholders()
        
        self.actor, self.critic, self._target_actor, self._target_critic = self._create_main_target_actor_critic()

        self.priorities, self.actor_loss, self.critic_loss = self._loss()
    
        self.opt_op = self._optimize_op(self.actor_loss, self.critic_loss)

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_loss()

    def _setup_env_placeholders(self):
        env_info = {}
        with tf.name_scope('placeholders'):
            env_info['state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
            env_info['action'] = tf.placeholder(tf.float32, shape=(None, self.action_size), name='action')
            env_info['next_state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='next_state')
            env_info['rewards'] = tf.placeholder(tf.float32, shape=(None, self.n_steps, 1), name='rewards')
            env_info['done'] = tf.placeholder(tf.uint8, shape=(None, 1), name='done')
            env_info['steps'] = tf.placeholder(tf.uint8, shape=(None, 1), name='steps')
            if self._buffer_type != 'uniform':
                env_info['IS_ratio'] = tf.placeholder(tf.float32, shape=(self.batch_size), name='importance_sampling_ratio')
        
        return env_info

    def _create_main_target_actor_critic(self):
        # main actor-critic
        actor, critic = self._create_actor_critic(is_target=False, is_double=self._double)
        # target actor-critic
        target_actor, target_critic = self._create_actor_critic(is_target=True, is_double=self._double)

        return actor, critic, target_actor, target_critic
        
    def _create_actor_critic(self, is_target=False, is_double=False):
        log_tensorboard = False if is_target else self._log_tensorboard

        scope_name = 'target' if is_target else 'main'
        scope_prefix = self.name + '/' + scope_name
        log_params = True if self._log_params and not is_target else False
        with tf.variable_scope(scope_name, reuse=self._reuse):
            actor = Actor('actor', self._args['actor'], self.env_info, self.action_size, 
                          reuse=self._reuse, log_tensorboard=log_tensorboard, 
                          is_target=is_target, 
                          scope_prefix=scope_prefix, log_params=log_params)

            critic = (DoubleCritic if is_double else Critic)('critic', self._args['critic'],
                                                             self.env_info, self.action_size, actor.action,
                                                             reuse=self._reuse, log_tensorboard=log_tensorboard,
                                                             is_target=is_target, 
                                                             distributional=self._distributional,
                                                             scope_prefix=scope_prefix, log_params=log_params)
        
        return actor, critic

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                Q_with_actor = self.critic.Q1_with_actor if self._double else self.critic.Q_with_actor
                actor_loss = tf.negative(tf.reduce_mean(Q_with_actor), name='actor_loss')

            with tf.name_scope('critic_loss'):
                if self._distributional and self._double:
                    priorities, critic_loss = self._double_distributional_critic_loss()
                elif self._distributional:
                    priorities, critic_loss = self._distributional_critic_loss()
                elif self._double:
                    priorities, critic_loss = self._double_critic_loss()
                else:
                    priorities, critic_loss = self._plain_critic_loss()

        return priorities, actor_loss, critic_loss

    def _double_distributional_critic_loss(self):
        v_min, v_max = self.critic.v_min, self.critic.v_max
        delta_z, Z_support = self.critic.delta_z, self.critic.Z_support
        Z1_probs, Z2_probs = self.critic.Z1_probs, self.critic.Z2_probs
        target_Z_probs_with_actor = self._target_critic.Z_probs_with_actor

        TZ = self._n_step_target(Z_support, stop_gradient=False)
        TZ = tf.expand_dims(tf.clip_by_value(TZ, v_min, v_max), axis=1)
        target_Z_probs_with_actor = tf.expand_dims(target_Z_probs_with_actor, axis=1)
        PhiTZ = tf.clip_by_value(1 - tf.abs(TZ - Z_support[None, :, None]) / delta_z, 0, 1) * target_Z_probs_with_actor
        self.PhiTZ = tf.stop_gradient(tf.reduce_sum(PhiTZ, axis=2), name='PhiTZ')

        # cross entropy
        cross_entropy1 = tf.negative(tf.reduce_sum(self.PhiTZ * tf.log(tf.minimum(1e-8, Z1_probs)), axis=1), name='cross_entropy1')
        cross_entropy2 = tf.negative(tf.reduce_sum(self.PhiTZ * tf.log(tf.minimum(1e-8, Z2_probs)), axis=1), name='cross_entropy2')

        priorities = tf.divide(cross_entropy1 + cross_entropy2, 2., name='priorities')

        critic_loss = self._average_critic_loss(priorities)

        return priorities, critic_loss

    def _distributional_critic_loss(self):
        v_min, v_max = self.critic.v_min, self.critic.v_max
        delta_z, Z_support = self.critic.delta_z, self.critic.Z_support
        Z_probs, target_Z_probs_with_actor = self.critic.Z_probs, self._target_critic.Z_probs_with_actor
        epsilon = 1e-5  # used to avoid the edge case where Z_probs is zero

        TZ = self._n_step_target(Z_support, stop_gradient=False)     # (batch_size, n_atoms)
        TZ = tf.expand_dims(tf.clip_by_value(TZ, v_min, v_max), axis=1)     # (batch_size, 1, n_atoms)
        target_Z_probs_with_actor = tf.expand_dims(target_Z_probs_with_actor, axis=1)   # (batch_size, 1, n_atoms)
        PhiTZ = (tf.clip_by_value(1 - tf.abs(TZ - Z_support[None, :, None]) / delta_z, 0, 1) 
                 * target_Z_probs_with_actor )  # (batch_size, n_atoms, n_atoms)
        self.PhiTZ = tf.stop_gradient(tf.reduce_sum(PhiTZ, axis=2), name='PhiTZ')
        
        # cross entropy
        priorities = tf.negative(tf.reduce_sum(self.PhiTZ * tf.log(tf.minimum(epsilon, Z_probs)), axis=1), name='priorities')

        critic_loss = self._average_critic_loss(priorities)

        return priorities, critic_loss

    def _double_critic_loss(self):
        target_Q = self._n_step_target(self._target_critic.Q_with_actor)
        
        TD_error1 = tf.abs(target_Q - self.critic.Q1, name='TD_error1')
        TD_error2 = tf.abs(target_Q - self.critic.Q2, name='TD_error2')
        priorities = tf.divide(TD_error1 + TD_error2, 2., name='priorities')

        if self._critic_loss == 'huber':
            TD_squared = huber_loss(TD_error1) + huber_loss(TD_error2)
        else:
            TD_squared = tf.square(TD_error1) + tf.square(TD_error2)

        critic_loss = self._average_critic_loss(TD_squared)

        return priorities, critic_loss
        
    def _plain_critic_loss(self):
        target_Q = self._n_step_target(self._target_critic.Q_with_actor)
        
        TD_error = tf.abs(target_Q - self.critic.Q, name='TD_error')
        priorities = tf.identity(TD_error, name='priorities')

        if self._critic_loss == 'huber':
            TD_squared = huber_loss(TD_error)
        else:
            TD_squared = tf.square(TD_error)

        critic_loss = self._average_critic_loss(TD_squared)
        
        return priorities, critic_loss

    def _average_critic_loss(self, loss):
        weighted_loss = loss if self._buffer_type == 'uniform' else self.env_info['IS_ratio'] * loss
        
        critic_loss = tf.reduce_mean(weighted_loss, name='critic_loss')

        return critic_loss

    def _n_step_target(self, n_step_value, stop_gradient=True):
        rewards_sum = tf.reduce_sum(self.env_info['rewards'], axis=1)
        n_step_gamma = tf.pow(self.gamma, tf.cast(self.env_info['steps'], tf.float32)) 
        n_step_target = tf.add(rewards_sum, n_step_gamma
                                            * tf.cast(1 - self.env_info['done'], tf.float32)
                                            * n_step_value, name='target_Q')
        if stop_gradient:
            return tf.stop_gradient(n_step_target)
        else:
            return n_step_target

    def _optimize_op(self, actor_loss, critic_loss):
        with tf.variable_scope('learn_steps', reuse=self._reuse):
            self.learn_steps = tf.get_variable('learn_steps', shape=[], 
                                               initializer=tf.constant_initializer(), trainable=False)
            step_op = tf.assign(self.learn_steps, self.learn_steps + 1, name='update_learn_steps')

        with tf.variable_scope('optimizer', reuse=self._reuse):
            actor_opt_op, _ = self.actor._optimization_op(actor_loss)
            critic_opt_op, _ = self.critic._optimization_op(critic_loss)

            with tf.control_dependencies([step_op]):
                opt_op = tf.group(actor_opt_op, critic_opt_op)

        return opt_op

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self._target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

    def _learn(self):
        def env_feed_dict():
            if self._buffer_type == 'uniform':
                feed_dict = {}
            else:
                feed_dict = {self.env_info['IS_ratio']: IS_ratios}

            feed_dict.update({
                self.env_info['state']: states,
                self.env_info['action']: actions,
                self.env_info['rewards']: rewards,
                self.env_info['next_state']: next_states,
                self.env_info['done']: dones,
                self.env_info['steps']: steps
            })

            return feed_dict

        IS_ratios, saved_exp_ids, (states, actions, rewards, next_states, dones, steps) = self.buffer.sample()

        feed_dict = env_feed_dict()

        # update the main networks
        if self._log_tensorboard:
            priorities, learn_steps, _, summary = self.sess.run([self.priorities, self.learn_steps, self.opt_op, self.graph_summary], feed_dict=feed_dict)
            if learn_steps % 100 == 0:
                self.writer.add_summary(summary, learn_steps)
                self.save()
        else:
            priorities, _ = self.sess.run([self.priorities, self.opt_op], feed_dict=feed_dict)

        if self._buffer_type != 'uniform':
            self.buffer.update_priorities(priorities, saved_exp_ids)

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
                    tf.summary.histogram('IS_ratios_', self.env_info['IS_ratio'])
                    tf.summary.scalar('IS_ratio_', tf.reduce_mean(self.env_info['IS_ratio']))

            with tf.variable_scope('loss', reuse=self._reuse):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('critic_loss_', self.critic_loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.critic.Q_with_actor))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.critic.Q_with_actor))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.critic.Q_with_actor))
            