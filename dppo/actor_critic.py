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
                 actions_ph,
                 advantages_ph,
                 env,
                 scope_prefix,
                 reuse=None):
        self.env = env

        self.actions_ph = actions_ph
        self.advantages_ph = advantages_ph
        super().__init__(name, args, graph, observations_ph, scope_prefix, reuse)

    """ Implementation """
    def _build_graph(self, **kwargs):
        output = self._network(self.observations_ph, 
                               self.env.is_action_discrete)

        self.action_distribution = self.env.action_dist_type(output)

        self.action = tf.squeeze(self.action_distribution.sample(), name='action')
        self.neglogpi = self.action_distribution.neglogp(self.actions_ph)

        self.loss = self._loss_func(self.neglogpi, self.advantages_ph)
        
    def _network(self, observation, discrete):
        x = observation
        x = self._dense(x, 256, kernel_initializer=tf_utils.kaiming_initializer())
        x = self._dense_resnet_norm_activation(x, 256)

        output_name = ('action_logits' if discrete else 'action_mean')
        x = self._dense(x, self.env.action_dim, name=output_name)

        if discrete:
            return x
        else:
            logstd = tf.get_variable('action_std', [self.env.action_dim], tf.float32)
            return x, logstd

    def _loss_func(self, neglogpi, advantages):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(neglogpi * advantages, name='actor_loss')

        return loss


class Critic(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 observations_ph, 
                 targetV_ph,
                 scope_prefix, 
                 reuse=None):
        self.loss_func = self._loss_func(args['loss_type'])
        self.targetV_ph = targetV_ph
        super().__init__(name, args, graph, observations_ph, scope_prefix, reuse)

    """ Implementation """
    def _build_graph(self, **kwargs):
        self.V = self._network(self.observations_ph)

        self.loss = self._loss(self.V, self.targetV_ph)

    def _network(self, observation):
        x = observation
        x = self._dense(x, 256, kernel_initializer=tf_utils.kaiming_initializer())
        x = self._dense_resnet_norm_activation(x, 256)
        x = self._dense_norm_activation(x, 256)
        x = self._dense(x, 1)

        x = tf.squeeze(x, name='V')

        return x

    def _loss(self, V, returns):
        with tf.name_scope('loss'):
            TD_error = returns - V
            losses = self.loss_func(TD_error)

            loss = tf.reduce_mean(losses, name='critic_loss')

        return loss

    def _loss_func(self, loss_type):
        if loss_type == 'huber':
            return huber_loss
        elif loss_type == 'mse':
            return tf.square
        else:
            raise NotImplementedError
