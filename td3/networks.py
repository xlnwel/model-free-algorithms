import tensorflow as tf

from basic_model.basic_nets import Base


class Actor(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state, 
                 action_dim, 
                 reuse=False, 
                 scope_prefix='', 
                 log_tensorboard=False, 
                 log_params=False):
        self.action_dim = action_dim
        self._noisy_sigma = args['noisy_sigma']
        super().__init__(name, 
                         args, 
                         graph,
                         state, 
                         reuse=reuse, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.action = self._deterministic_policy_net(self.state, self.args['units'], self.action_dim, 
                                                    self._noisy_sigma, reuse=self.reuse)


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
                 reuse=False, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.action = action
        self.actor_action = actor_action
        self.action_dim = action_dim
        
        super().__init__(name, 
                         args, 
                         graph, 
                         state,
                         reuse=reuse, 
                         scope_prefix=scope_prefix, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q, self.Q_with_actor = self._build_net('Qnet')

    def _build_net(self, name):
        with tf.variable_scope(name, reuse=self.reuse):
            Q = self._Q_net(self.state, self.args['units'], self.action, self.action_dim, self.reuse)
            Q_with_actor = self._Q_net(self.state, self.args['units'], self.actor_action, self.action_dim, True)

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
                 reuse=False, 
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
                         reuse=reuse, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q1, self.Q1_with_actor = self._build_net(name='Qnet1')
        self.Q2, self.Q2_with_actor = self._build_net(name='Qnet2')
        self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')
