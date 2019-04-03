import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
from basic_model.basic_nets import Base


class Actor(Base):
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
        x = state
        with tf.variable_scope(name):
            for i, u in enumerate(units):
                layer = self.dense_norm_activation if i < self.args['n_fc'] else self.noisy_norm_activation
                x = layer(x, u, norm=self.norm)
            x = self.noisy(x, action_dim, sigma=noisy_sigma)
            x = tf.tanh(x, name='action')

        return x


class Critic(Base):
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
        with tf.variable_scope(name):
            Q = self._Q_net(self.state, self.args['units'], self.action, 
                            self.action_dim, self.norm, False)
            Q_with_actor = self._Q_net(self.state, self.args['units'], self.actor_action, 
                                       self.action_dim, self.norm, True)

        return Q, Q_with_actor
        

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
