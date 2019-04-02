import tensorflow as tf

from basic_model.basic_nets import Base


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
    
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
        self.action_dim = env.action_dim

        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        mean, logstd = self._stochastic_policy_net(self.state, self.args['units'], self.action_dim)

        self.action_distribution = self.env.action_dist_type((mean, logstd))

        action = self.action_distribution.sample()
        logpi = self.action_distribution.logp(action)

        # Enforcing action bound
        self.action = tf.tanh(action)
        self.logpi = logpi - tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - self.action**2, l=0, u=1) + 1e-6), axis=1)
        # make sure actions are in correct range
        self.action *= self.env.action_high[0]

    def _stochastic_policy_net(self, state, units, action_dim):
        x = state
        with tf.variable_scope('policy_net'):
            for u in units:
                x = self.dense_norm_activation(x, u, normalization=None)

            mean = self.dense(x, action_dim)

            # logstd computation follows the implementation of OpenAI Spinning Up
            logstd = self.dense_norm_activation(x, action_dim, activation=tf.tanh)
            LOG_STD_MAX = 2
            LOG_STD_MIN = -20
            logstd = LOG_STD_MIN + .5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)

        return mean, logstd


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
                 action_dim, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.action = action
        self.actor_action = actor_action
        self.action_dim = action_dim
        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        self.Q1 = self._Q_net(self.state, self.args['units'], self.action, 
                              self.action_dim, False, name='Qnet1')
        self.Q2 = self._Q_net(self.state, self.args['units'], self.action, 
                              self.action_dim, False, name='Qnet2')
        self.Q1_with_actor = self._Q_net(self.state, self.args['units'], self.actor_action, 
                                         self.action_dim, True, name='Qnet1')
        self.Q2_with_actor = self._Q_net(self.state, self.args['units'], self.actor_action, 
                                         self.action_dim, True, name='Qnet2')
        self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')
