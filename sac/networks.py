import numpy as np
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
                 reuse=False,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.env = env
        self.action_dim = env.action_dim
        super().__init__(name, 
                         args, 
                         graph, 
                         state,
                         reuse=reuse, 
                         scope_prefix=scope_prefix, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        dist_params = self._stochastic_policy_net(self.state, self.action_dim, 
                                            self._reuse, discrete=self.env.is_action_discrete)

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
                 reuse=False, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.next_state = next_state
        self.tau = args['tau']
        super().__init__(name, 
                         args, 
                         graph, 
                         state,
                         reuse=reuse, 
                         scope_prefix=scope_prefix, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    @property
    def main_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/' + 'main')
    
    @property
    def target_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/' + 'target')

    def _build_graph(self):
        self.V = self._V_net(self.state, self.reuse, 'main')
        self.target_V = self._V_net(self.next_state, self.reuse, 'target')  # target V use next state as input

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
                 policy_action,
                 action_dim, 
                 reuse=False, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.action = action
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
        units = [500, 500, 500]
        self.Q1 = self._Q_net(self.state, units, self.action, self.action_dim, self.reuse, name='Qnet1')
        self.Q1_with_actor = self._Q_net(self.state, units, self.action, self.action_dim, True, name='Qnet1')
        self.Q2 = self._Q_net(self.state, self.action, units, self.action_dim, self.reuse, name='Qnet2')
        self.Q2_with_actor = self._Q_net(self.state, units, self.action, self.action_dim, True, name='Qnet2')
        self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')
