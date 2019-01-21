import numpy as np
import tensorflow as tf
import gym

from basic_model.model import Module
import utils.tf_utils as tf_utils

class Base(Module):
    def __init__(self,
                 name,
                 args,
                 graph,
                 observation_ph,
                 scope_prefix,
                 reuse):
        self.observation_ph = observation_ph
        self._variable_scope = (scope_prefix + '/' 
                                if scope_prefix != '' and not scope_prefix.endswith('/') 
                                else scope_prefix) + name

        super().__init__(name, args, graph, reuse=reuse)


class Actor(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 observation_ph,
                 observation_dim,
                 action_dim,
                 is_action_discrete,
                 scope_prefix,
                 reuse=None):
        self._is_action_discrete = is_action_discrete
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self._noisy_sigma = args['noisy_sigma']

        super().__init__(name, args, graph, observation_ph, scope_prefix, reuse)
    
    """ Implementation """
    def _build_graph(self):
        output = self._network(self.observation_ph, self._noisy_sigma, 
                               self._is_action_discrete)
        if self._is_action_discrete:
            self.logits = output
        else:
            self.mean, self.logstd = output

    def _network(self, observation, noisy_sigma, discrete):
        x = self._dense(observation, 512)
        # x = self._dense_resnet(x, 512)
        x = self._dense_resnet_norm_activation(x, 512)
        x = self._noisy_norm_activation(x, 256, sigma=self._noisy_sigma)

        output_name = ('action_logits' if discrete else 'action_mean')
        x = self._noisy(x, self.action_dim, sigma=self._noisy_sigma, name=output_name)

        if discrete:
            return x
        else:
            logstd = tf.get_variable('action_std', [self.action_dim], tf.float32)
            return x, logstd


class Critic(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 observation_ph, 
                 scope_prefix, 
                 reuse=None):
        super().__init__(name, args, graph, observation_ph, scope_prefix, reuse)
        
    """ Implementation """
    def _build_graph(self):
        self.V = self._network(self.observation_ph, self._reuse)

    def _network(self, observation, reuse):
        x = self._dense(observation, 512, kernel_initializer=tf_utils.kaiming_initializer())
        # x = self._dense_resnet(x, 512)
        x = self._dense_resnet_norm_activation(x, 512)
        x = self._dense_norm_activation(x, 256)
        x = self._dense(x, 1, name='V')

        return x