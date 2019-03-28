import tensorflow as tf

from basic_model.basic_nets import Base


class SoftPolicy(Base):
    """ Interface """
    def __init__(self,
                 name,
                 args,
                 graph,
                 state,
                 env,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.env = env
        self.action_space = env.action_space
        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        dist_params = self._stochastic_policy_net(self.state, self.args['units'], self.action_space, 
                                                  discrete=self.env.is_action_discrete)

        self.action_distribution = self.env.action_dist_type(dist_params)

        self.action = self.action_distribution.sample()


class SoftV(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state,
                 next_state,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.next_state = next_state
        self.tau = args['tau']
        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    @property
    def main_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/main')
    
    @property
    def target_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/target')

    def _build_graph(self):
        self.V = self._V_net(self.state, self.args['units'], 'main')
        self.target_V = self._V_net(self.next_state, self.args['units'], 'target')  # target V use next state as input

        self.init_target_op, self.update_target_op = self._target_net_ops()

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op   


class SoftQ(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state,
                 action,
                 actor_action,
                 action_space, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.action = action
        self.actor_action = actor_action
        self.action_space = action_space
        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q1 = self._Q_net(self.state, self.args['units'], self.action, self.action_space, False, name='Qnet1')
        self.Q1_with_actor = self._Q_net(self.state, self.args['units'], self.actor_action, self.action_space, True, name='Qnet1')
        self.Q2 = self._Q_net(self.state, self.args['units'], self.action, self.action_space, False, name='Qnet2')
        self.Q2_with_actor = self._Q_net(self.state, self.args['units'], self.actor_action, self.action_space, True, name='Qnet2')
        self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')
