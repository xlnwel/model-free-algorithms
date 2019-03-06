import numpy as np
import tensorflow as tf
import gym

from basic_model.basic_nets import Base
from utility import tf_utils, tf_distributions
from utility.losses import huber_loss


class Actor(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state,
                 old_neglogpi_ph,
                 advantage_ph,
                 entropy_coef_ph,
                 env,
                 scope_prefix,
                 reuse=None):
        self.env = env
        self.clip_range = args['clip_range']

        self.old_neglogpi_ph = old_neglogpi_ph
        self.advantage_ph = advantage_ph
        self.entropy_coef_ph = entropy_coef_ph

        super().__init__(name, args, graph, state, scope_prefix, reuse)

    """ Implementation """
    def _build_graph(self):
        output = self._network(self.state, self.env.is_action_discrete)

        self.action_distribution = self.env.action_dist_type(output)

        self.action = self.action_distribution.sample()
        # self.action = tf.clip_by_value(self.action_distribution.sample(), self.env.action_low, self.env.action_high)
        self.neglogpi = self.action_distribution.neglogp(tf.stop_gradient(self.action))

        self.ppo_loss, self.entropy, self.loss = self._loss_func(self.neglogpi, self.old_neglogpi_ph, self.advantage_ph, self.clip_range)

    def _network(self, observation, discrete):
        x = observation
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.nn.relu)
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.nn.relu)

        output_name = ('action_logits' if discrete else 'action_mean')
        x = self._dense(x, self.env.action_dim, name=output_name)

        if discrete:
            return x
        else:
            logstd = tf.get_variable('action_logstd', [self.env.action_dim], tf.float32)
            return x, logstd

    def _loss_func(self, neglogpi, old_neglogpi, advantages, clip_range):
        with tf.name_scope('loss'):
            ratio = tf.exp(old_neglogpi - neglogpi)
            clipped_ratio = tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
            loss1 = -ratio * advantages
            loss2 = -clipped_ratio * advantages
            ppo_loss = tf.reduce_mean(tf.maximum(loss1, loss2, name='ppo_loss'))
            entropy = tf.reduce_mean(self.entropy_coef_ph * self.action_distribution.entropy(), name='entropy_loss')
            
            loss = tf.subtract(ppo_loss, entropy, name='loss')

        return ppo_loss, entropy, loss


class Critic(Base):
    def __init__(self,
                 name,
                 args,
                 graph,
                 state,
                 return_ph,
                 scope_prefix,
                 reuse=None):
        self.loss_func = self._loss_func(args['loss_type'])
        self.return_ph = return_ph
        super().__init__(name, args, graph, state, scope_prefix, reuse)

    """ Implementation """
    def _build_graph(self):
        self.V = self._network(self.state)

        self.loss = self._loss(self.V, self.return_ph)

    def _network(self, observation):
        x = observation
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.nn.relu)
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.nn.relu)

        x = self._dense(x, 1)

        x = tf.squeeze(x, name='V')

        return x

    def _loss(self, V, returns):
        with tf.name_scope('loss'):
            TD_error = returns - V
            losses = self.loss_func(TD_error)

            loss = .5 * tf.reduce_mean(losses, name='critic_loss')

        return loss

    def _loss_func(self, loss_type):
        if loss_type == 'huber':
            return huber_loss
        elif loss_type == 'mse':
            return tf.square
        else:
            raise NotImplementedError
