import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from basic_model.model import Module
import utility.tf_utils as tf_utils

class Base(Module):
    def __init__(self, name, args, env_info, reuse=False, 
                 log_tensorboard=True, is_target=False, 
                 trainable=True, scope_prefix='', log_params=False):
        self.observations = env_info['observations'] if not is_target else env_info['next_state']
        self._trainable = trainable
        self._variable_scope = (scope_prefix + '/' 
                                if scope_prefix != '' and not scope_prefix.endswith('/') 
                                else scope_prefix) + name
        
        super().__init__(name, args, reuse=reuse, log_tensorboard=log_tensorboard, log_params=log_params)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self._variable_scope)

class Actor(Base):
    """ Interface """
    def __init__(self, name, args, env_info, action_size, 
                 reuse=False, log_tensorboard=True, 
                 is_target=False, trainable=True, 
                 scope_prefix='', log_params=False):
        self.action_size = action_size
        self._noisy_sigma = args['noisy_sigma']
        super().__init__(name, args, env_info, reuse=reuse, 
                         log_tensorboard=log_tensorboard, 
                         is_target=is_target, trainable=trainable, 
                         scope_prefix=scope_prefix, log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.action = self._actor(self.observations)
        
    def _actor(self, observations):
        with tf.variable_scope('net', reuse=self._reuse):
            x = self._dense(observations, 512)
            # x = self._dense_resnet(x, 512)
            x = self._dense_resnet_norm_activation(x, 512)
            x = self._noisy_norm_activation(x, 256, sigma=self._noisy_sigma)
            x = self._noisy(x, self.action_size, sigma=self._noisy_sigma)
            x = tf.tanh(x, name='action')

        return x


class Critic(Base):
    """ Interface """
    def __init__(self, name, args, env_info, 
                 action_size, actor_action, 
                 reuse=False, log_tensorboard=True,
                 is_target=False, trainable=True,
                 scope_prefix='', log_params=False):
        self.action_size = action_size
        self.action = env_info['action']
        self.actor_action = actor_action

        super().__init__(name, args, env_info, reuse=reuse, 
                         log_tensorboard=log_tensorboard, 
                         is_target=is_target, trainable=trainable, 
                         scope_prefix=scope_prefix, log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q, self.Q_with_actor = self._build_plain_net()

    def _build_plain_net(self, name=None):
        if name:
            with tf.variable_scope(name, reuse=self._reuse):
                Q = self._plain_critic(self.observations, self.action, self._reuse)
                Q_with_actor = self._plain_critic(self.observations, self.actor_action, True)
        else:
            Q = self._plain_critic(self.observations, self.action, self._reuse)
            Q_with_actor = self._plain_critic(self.observations, self.actor_action, True)

        return Q, Q_with_actor

    def _plain_critic(self, observations, action, reuse):
        self._reset_counter('dense_resnet')
        
        with tf.variable_scope('plain_net', reuse=reuse):
            x = self._dense(observations, 512 - self.action_size, kernel_initializer=tf_utils.kaiming_initializer())
            x = tf.concat([x, action], 1)
            # x = self._dense_resnet(x, 512)
            x = self._dense_resnet_norm_activation(x, 512)
            x = self._dense_norm_activation(x, 256)
            x = self._dense(x, 1, name='Q')

        return x
        
class DoubleCritic(Critic):
    """ Interface """
    def __init__(self, name, args, env_info, 
                 action_size, actor_action, 
                 reuse=False, log_tensorboard=True, 
                 is_target=False, trainable=True, 
                 scope_prefix='', log_params=False):
        super().__init__(name, args, env_info, action_size, actor_action,
                         reuse=reuse, log_tensorboard=log_tensorboard,
                         is_target=is_target, trainable=trainable,
                         scope_prefix=scope_prefix, log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q1, self.Q1_with_actor = self._build_plain_net(name='net1')
        self.Q2, self.Q2_with_actor = self._build_plain_net(name='net2')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')
