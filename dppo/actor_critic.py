import numpy as np
import tensorflow as tf
import gym

from basic_model.model import Module
from utils import tf_utils, tf_distributions
from utils.losses import huber_loss

class Base(Module):
    def __init__(self,
                 name,
                 args,
                 graph,
                 observations_ph,
                 scope_prefix,
                 reuse):
        self.observations_ph = observations_ph
        self._variable_scope = (scope_prefix + '/' 
                                if scope_prefix != '' and not scope_prefix.endswith('/') 
                                else scope_prefix) + name

        super().__init__(name, args, graph, reuse=reuse)


class Actor(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 observations_ph,
                 advantage_ph,
                 env,
                 scope_prefix,
                 reuse=None):
        self.env = env
        self.epsilon = args['epsilon']
        self.entropy_coef = args['entropy_coef']

        self.advantage_ph = advantage_ph
        super().__init__(name, args, graph, observations_ph, scope_prefix, reuse)

    """ Implementation """
    def _build_graph(self, **kwargs):
        self.old_neglogpi_ph = tf.placeholder(tf.float32, [None, 1], name='old_neglogpi')

        output = self._network(self.observations_ph, 
                               self.env.is_action_discrete)

        self.action_distribution = self.env.action_dist_type(output)

        self.action = self.action_distribution.sample()
        self.neglogpi = self.action_distribution.neglogp(tf.stop_gradient(self.action))

        self.loss = self._loss_func(self.neglogpi, self.old_neglogpi_ph, self.advantage_ph, self.epsilon)
        
    def _network(self, observation, discrete):
        x = observation
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.tanh)
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.tanh)

        output_name = ('action_logits' if discrete else 'action_mean')
        x = self._dense(x, self.env.action_dim, name=output_name)

        if discrete:
            return x
        else:
            logstd = tf.get_variable('action_std', [self.env.action_dim], tf.float32)
            return x, logstd

    def _loss_func(self, neglogpi, old_neglogpi, advantages, epsilon):
        with tf.name_scope('loss'):
        #     ratio = tf.exp(old_neglogpi - neglogpi)
        #     clipped_ratio = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon)
        #     objectvie1 = ratio * advantages
        #     objective2 = clipped_ratio * advantages
        #     loss = -tf.reduce_mean(tf.minimum(objectvie1, objective2, name='ppo_loss')
        #             + self.entropy_coef * self.action_distribution.entropy(), name='actor_loss')
            loss = tf.reduce_mean(neglogpi * advantages)

        return loss


class Critic(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 observations_ph,
                 return_ph,
                 scope_prefix,
                 reuse=None):
        self.loss_func = self._loss_func(args['loss_type'])
        self.return_ph = return_ph
        super().__init__(name, args, graph, observations_ph, scope_prefix, reuse)

    """ Implementation """
    def _build_graph(self, **kwargs):
        self.V = self._network(self.observations_ph)

        self.loss = self._loss(self.V, self.return_ph)

    def _network(self, observation):
        x = observation
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.tanh)
        x = self._dense_norm_activation(x, 64, normalization=None, activation=tf.tanh)

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
