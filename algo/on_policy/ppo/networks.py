import numpy as np
import tensorflow as tf
import gym

from basic_model.basic_nets import Base
from utility import tf_utils, tf_distributions


class ActorCritic(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 env_vec,
                 env_phs,
                 scope_prefix,
                 log_tensorboard=False,
                 log_params=False):
        self.env_vec = env_vec
        self.env_phs = env_phs
        self.clip_range = args['clip_range']
        self.use_rnn = args['use_rnn']

        super().__init__(name,
                         args,
                         graph,
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)
        
    """ Implementation """
    def _build_graph(self):
        x = self.env_phs['state']
        
        if self.args['common']:
            x = self._common_dense(x, self.args['common_dense_units'], reshape_for_rnn=self.use_rnn)
            x, self.initial_state, self.final_state = self._common_lstm(x, self.args['common_lstm_units'])
            actor_output = self._policy_head(x, self.args['actor_units'], self.env_vec.action_dim, 
                                            discrete=self.env_vec.is_action_discrete)
            self.V = self._V_head(x, self.args['critic_units'])
        else:
            # independent actor & critic networks
            actor_output = self._policy_net(x, self.args['actor_units'], 
                                            self.env_vec.action_dim, 
                                            discrete=self.env_vec.is_action_discrete)

            self.V = self._V_net(x, self.args['critic_units'])
            if self.use_rnn:
                self.initial_state = [*self.actor_init_state, *self.critic_init_state]
                self.final_state = [*self.actor_final_state, *self.critic_final_state]

        self.action_distribution = self.env_vec.action_dist_type(actor_output)
        self.action = self.action_distribution.sample()
        self.logpi = self.action_distribution.logp(tf.stop_gradient(self.action))

        # losses
        self.ppo_loss, self.entropy, self.approx_kl, self.clipfrac, self.V_loss, self.loss = self._loss()

        # optimizer
        self.optimizer, self.opt_step, self.grads_and_vars, self.opt_op = self._optimize(self.loss)

    def _common_dense(self, x, units, reshape_for_rnn=True, name='common_dense'):
        with tf.variable_scope(name):
            for u in units:
                x = self.dense_norm_activation(x, u)
            x = tf.reshape(x, (self.env_vec.n_envs, -1, u))

        return x

    def _common_lstm(self, x, units, name='common_lstm'):
        init_state_list, final_state_list = [], []
        with tf.variable_scope(name):
            for u in units:
                x, (init_state, final_state) = self.lstm(x, u, return_sequences=True)
                init_state_list += init_state
                final_state_list += final_state

            x = tf.reshape(x, (-1, u))

        return x, init_state_list, final_state_list

    def _policy_head(self, x, units, action_dim, discrete=False, name='policy_head'):
        output_name = 'action_logits' if discrete else 'action_mean'
        with tf.variable_scope(name):
            x = self._feedforward_net(x, units, action_dim, output_name)

            if discrete:
                return x
            else:
                logstd = tf.get_variable('action_logstd', [action_dim], tf.float32)
                
                return x, logstd

    def _V_head(self, x, units, name='V_head'):
        with tf.variable_scope(name):
            x = self._feedforward_net(x, units, 1, name)

        return x

    def _policy_net(self, x, units, action_dim, discrete=False, name='policy_net'):
        with tf.variable_scope(name):
            x = self._common_dense(x, self.args['common_dense_units'], 
                                   reshape_for_rnn=self.use_rnn, name='dense')
            if self.use_rnn:
                x, self.actor_init_state, self.actor_final_state = self._common_lstm(x, self.args['common_lstm_units'], name='lstm')

            output = self._policy_head(x, self.args['actor_units'], self.env_vec.action_dim, 
                                        discrete=self.env_vec.is_action_discrete)
            
            return output

    def _V_net(self, x, units, name='V_net'):
        with tf.variable_scope(name):
            x = self._common_dense(x, self.args['common_dense_units'], reshape_for_rnn=self.use_rnn, name='dense')
            if self.use_rnn:
                x, self.critic_init_state, self.critic_final_state = self._common_lstm(x, self.args['common_lstm_units'], name='lstm')

            x = self._V_head(x, self.args['critic_units'])

        return x

    def _feedforward_net(self, x, units, output_dim, output_name):
        for u in units:
            x = self.dense_norm_activation(x, u)
        x = self.dense(x, output_dim, name=output_name)

        return x

    def _loss(self):
        with tf.name_scope('loss'):
            loss_info = self._policy_loss(self.logpi, self.env_phs['old_logpi'], 
                                        self.env_phs['advantage'], self.clip_range, 
                                        self.action_distribution.entropy())
            ppo_loss, entropy, approx_kl, clipfrac = loss_info
            V_loss = self._value_loss(self.V, self.env_phs['return'], self.env_phs['value'], self.clip_range)
            loss = ppo_loss - self.env_phs['entropy_coef'] * entropy + V_loss * self.args['value_coef']

        return ppo_loss, entropy, approx_kl, clipfrac, V_loss, loss

    def _policy_loss(self, logpi, old_logpi, advantages, clip_range, entropy):
        with tf.name_scope('policy_loss'):
            ratio = tf.exp(logpi - old_logpi)
            loss1 = -advantages * ratio
            loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
            ppo_loss = tf.reduce_mean(tf.maximum(loss1, loss2), name='ppo_loss')
            entropy = tf.reduce_mean(entropy, name='entropy_loss')
            # debug stats: KL between old and current policy and fraction of data being clipped
            approx_kl = .5 * tf.reduce_mean(tf.square(old_logpi - logpi))
            clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32))

        return ppo_loss, entropy, approx_kl, clipfrac

    def _value_loss(self, V, returns, value, clip_range):
        with tf.name_scope('value_loss'):
            if self.args['value_loss_type'] == 'mse':
                loss = .5 * tf.reduce_mean((returns - V)**2, name='critic_loss')
            elif self.args['value_loss_type'] == 'clip':
                V_clipped = value + tf.clip_by_value(V - value, -clip_range, clip_range)
                loss1 = (V - returns)**2
                loss2 = (V_clipped - returns)**2
                loss = .5 * tf.reduce_mean(tf.maximum(loss1, loss2))
            else:
                NotImplementedError

        return loss

    def _optimize(self, loss):
        with tf.name_scope('optimization'):
            optimizer, opt_step = self._adam_optimizer(opt_step=True)
            grads_and_vars = self._compute_gradients(loss, optimizer)
            opt_op = self._apply_gradients(optimizer, grads_and_vars, opt_step)

        return optimizer, opt_step, grads_and_vars, opt_op

