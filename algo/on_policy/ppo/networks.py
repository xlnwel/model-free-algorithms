import numpy as np
import tensorflow as tf
import gym

from basic_model.basic_nets import Base
from utility import tf_utils, tf_distributions
from utility.utils import pwc

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

        super().__init__(name,
                         args,
                         graph,
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)
        
    """ Implementation """
    def _build_graph(self):
        x = self.env_phs['state']

        x = self._common_layers(x)
        
        # actor & critic networks
        actor_output = self._stochastic_policy_net(x, self.args['actor_units'], self.env_vec.action_space, 
                                                   discrete=self.env_vec.is_action_discrete)
        self.V = self._V_net(x, self.args['critic_units'])

        self.action_distribution = self.env_vec.action_dist_type(actor_output)
        self.action = self.action_distribution.sample()
        self.logpi = self.action_distribution.logp(tf.stop_gradient(self.action))

        # losses
        self.ppo_loss, self.entropy, self.approx_kl, self.clipfrac, self.V_loss, self.loss = self._loss()

        # optimizer
        self.optimizer, self.opt_step, self.grads_and_vars, self.opt_op = self._optimize(self.loss)

    def _common_layers(self, x):
        # common network shared by actor and critic
        shared_fc_units = self.args['shared_fc_units'] if 'shared_fc_units' in self.args else None
        lstm_units = self.args['lstm_units'] if 'lstm_units' in self.args else None

        with tf.variable_scope('common'):
            if shared_fc_units:
                if not isinstance(shared_fc_units, list):
                    shared_fc_units = [shared_fc_units]
                for u in shared_fc_units:
                    x = self.dense_norm_activation(x, u)
            if lstm_units:
                x = tf.reshape(x, (self.env_vec.n_envs, -1, shared_fc_units))
                x, (self.initial_state, self.final_state) = self.lstm(x, lstm_units, return_sequences=True)
                x = tf.reshape(x, (-1, lstm_units))

        return x

    def _stochastic_policy_net(self, state, units, action_space, 
                               discrete=False, name='policy_net'):
        x = state
        with tf.variable_scope(name):
            for u in units:
                x = self.dense_norm_activation(x, u, normalization=None)
            output_name = ('action_logits' if discrete else 'action_mean')
            y = self.dense(x, action_space, name=output_name)

            if discrete:
                return y
            else:
                logstd = tf.get_variable('action_logstd', [action_space], tf.float32)
                
                return y, logstd

    def _loss(self):
        with tf.name_scope('loss'):
            loss_info = self._policy_loss(self.logpi, self.env_phs['old_logpi'], 
                                        self.env_phs['advantage'], self.clip_range, 
                                        self.action_distribution.entropy())
            ppo_loss, entropy, approx_kl, clipfrac = loss_info
            V_loss = self._value_loss(self.V, self.env_phs['return'], self.clip_range)
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

    def _value_loss(self, V, returns, clip_range):
        with tf.name_scope('value_loss'):
            if self.args['value_loss_type'] == 'mse':
                loss = .5 * tf.reduce_mean((returns - V)**2, name='critic_loss')
            elif self.args['value_loss_type'] == 'clip':
                V_clipped = self.env_phs['value'] + tf.clip_by_value(V - self.env_phs['value', -clip_range, clip_range])
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
