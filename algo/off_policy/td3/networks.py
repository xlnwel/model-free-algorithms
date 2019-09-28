import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from basic_model.model import Module


class Actor(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state, 
                 action_dim, 
                 scope_prefix='',
                 log_tensorboard=False, 
                 log_params=False):
        self.state = state
        self.action_dim = action_dim
        self.noisy_sigma = args['noisy_sigma']
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None
        super().__init__(name, 
                         args, 
                         graph,
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.action = self._deterministic_policy_net(self.state, self.args['units'], self.action_dim, 
                                                     self.noisy_sigma)

    def _deterministic_policy_net(self, state, units, action_dim, noisy_sigma, name='policy_net'):
        noisy_norm_activation = lambda x, u, norm: self.noisy_norm_activation(x, u, norm=norm, sigma=noisy_sigma)
        x = state
        with tf.variable_scope(name):
            for i, u in enumerate(units):
                layer = self.dense_norm_activation if i < len(units) - self.args['n_noisy']  else noisy_norm_activation
                x = layer(x, u, norm=self.norm)
            x = self.dense(x, action_dim)
            x = tf.tanh(x, name='action')

        return x


class Critic(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state,
                 action,
                 actor_action, 
                 action_dim,
                 scope_prefix='', 
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.action = action
        self.actor_action = actor_action
        self.action_dim = action_dim
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None
        
        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q, self.Q_with_actor = self._build_net('Qnet')

    def _build_net(self, name):
        Q_net = lambda action, reuse: self._q_net(self.state, self.args['units'], action, 
                                          self.norm, reuse)
        with tf.variable_scope(name):
            Q = Q_net(self.action, None)
            Q_with_actor = Q_net(self.actor_action, True)

        return Q, Q_with_actor
    
    def _q_net(self, state, units, action, norm, reuse, name='Q_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for i, u in enumerate(units):
                if i < 2:
                    x = tf.concat([x, action], 1)
                x = self.dense_norm_activation(x, u, norm=norm)

            x = self.dense(x, 1, name='Q')

        return x


class DoubleCritic(Critic):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state,
                 action,
                 actor_action, 
                 action_dim,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        super().__init__(name, 
                         args, 
                         graph, 
                         state,
                         action,
                         actor_action,
                         action_dim,
                         scope_prefix=scope_prefix, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q1, self.Q1_with_actor = self._build_net(name='Qnet1')
        self.Q2, self.Q2_with_actor = self._build_net(name='Qnet2')
        self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')
