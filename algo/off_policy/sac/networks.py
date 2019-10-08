import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from basic_model.model import Module


class SoftPolicy(Module):
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
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None
        self.noisy_sigma = args['noisy_sigma']

        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        mean, logstd, self.mean_det = self._stochastic_policy_net(self.state, self.args['units'], self.action_dim, self.norm, self.noisy_sigma)

        self.action_distribution = self.env.action_dist_type((mean, logstd))

        orig_action = self.action_distribution.sample()
        orig_logpi = self.action_distribution.logp(orig_action)

        # Enforcing action bound
        self.action, self.logpi = self._squash_correction(orig_action, orig_logpi)

    def _stochastic_policy_net(self, state, units, action_dim, norm, noisy_sigma, name='policy_net'):
        noisy_norm_activation = lambda x, u, norm: self.noisy_norm_activation(x, u, norm=norm, sigma=noisy_sigma)
        x = state
        with tf.variable_scope(name):
            for i, u in enumerate(units):
                layer = self.dense_norm_activation if i < len(units) - self.args['n_noisy']  else noisy_norm_activation
                x = layer(x, u, norm=norm)

            mean = self.dense(x, action_dim)

            # constrain logstd to be in range [LOG_STD_MIN, LOG_STD_MAX]
            logstd = self.dense(x, action_dim)
            logstd = tf.clip_by_value(logstd, -20., 2.)
        
        x_det = state
        with tf.variable_scope(name, reuse=True):
            for u in units:
                x_det = self.dense_norm_activation(x_det, u, norm=norm)

            mean_det = self.dense(x_det, action_dim)

        return mean, logstd, mean_det

    def _squash_correction(self, action, logpi):
        with tf.name_scope('squash'):
            action_new = tf.tanh(action)
            sub = 2 * tf.reduce_sum(tf.log(2.) + action - tf.nn.softplus(2 * action), axis=1)
            logpi -= sub

        return action_new, logpi

class SoftV(Module):
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
        self.polyak = args['polyak']
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None
        
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
        self.V = self._v_net(self.state, self.args['units'], self.norm, 'main')
        self.V_next = self._v_net(self.next_state, self.args['units'], self.norm, 'target')  # target V use next state as input

        self.init_target_op, self.update_target_op = self._target_net_ops()

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.polyak * v[0] + (1. - self.polyak) * v[1], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op   
    
    def _v_net(self, state, units, norm, name='V_net'):
        x = state
        with tf.variable_scope(name):
            for u in units:
                x = self.dense_norm_activation(x, u, norm=norm)
            x = self.dense(x, 1, name='V')

        return x


class SoftQ(Module):
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
        Q_net = lambda action, reuse, name: self._q_net(self.state, self.args['units'], action, 
                                                        self.norm, reuse, name=name)

        self.Q1 = Q_net(self.action, False, 'Qnet1')
        self.Q2 = Q_net(self.action, False, 'Qnet2')
        self.Q1_with_actor = Q_net(self.actor_action, True, 'Qnet1')
        self.Q2_with_actor = Q_net(self.actor_action, True, 'Qnet2')
        self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
        self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')

    def _q_net(self, state, units, action, norm, reuse, name='Q_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for i, u in enumerate(units):
                if i < 2:
                    x = tf.concat([x, action], 1)
                x = self.dense_norm_activation(x, u, norm=norm)

            x = self.dense(x, 1, name='Q')

        return x