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

        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        self.mean, logstd = self._stochastic_policy_net(self.state, self.args['units'], self.action_dim, self.norm)

        self.action_distribution = self.env.action_dist_type((self.mean, logstd))

        orig_action = self.action_distribution.sample()
        orig_logpi = self.action_distribution.logp(orig_action)

        # Enforcing action bound
        self.action, self.logpi = self._squash_correction(orig_action, orig_logpi)

    def _stochastic_policy_net(self, state, units, action_dim, norm):
        noisy_norm_activation = lambda x, u, norm: self.noisy_norm_activation(x, u, norm=norm, sigma=self.args['noisy_sigma'])
        x = state
        with tf.variable_scope('policy_net'):
            for i, u in enumerate(units):
                layer = self.dense_norm_activation if i < len(units) - self.args['n_noisy']  else noisy_norm_activation
                x = layer(x, u, norm=norm)

            mean = self.dense(x, action_dim)

            # constrain logstd to be in range [LOG_STD_MIN, LOG_STD_MAX]
            logstd = self.dense(x, action_dim)
            logstd = tf.clip_by_value(logstd, -20., 2.)

        return mean, logstd

    def _squash_correction(self, action, logpi):
        with tf.name_scope('squash'):
            action_new = tf.tanh(action)
            sub = 2 * tf.reduce_sum(tf.log(2.) + action - tf.nn.softplus(2 * action), axis=1, keepdims=True)
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

class Temperature(Module):
    def __init__(self,
                 name,
                 args,
                 graph,
                 state,
                 next_state,
                 policy,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        # next_* are used if the state value function is omitted
        self.state = state
        self.next_state = next_state
        self.action = policy.action
        # self.next_action = policy.next_action
        self.type = args['type']
        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        if self.type == 'simple':
            self.log_alpha, self.alpha = self._simple_alpha()
            # self.next_alpha = self.alpha    
        elif self.type == 'state':
            self.log_alpha, self.alpha = self._state_alpha(self.state)
            # _, self.next_alpha = self._state_alpha(self.next_state, reuse=True)
        elif self.type == 'state_action':
            self.log_alpha, self.alpha = self._state_action_alpha(self.state, self.action)
            # _, self.next_alpha = self._state_action_alpha(self.next_state, self.next_action, reuse=True)

    def _simple_alpha(self):
        with tf.variable_scope('temperature'):
            log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.)
            alpha = tf.exp(log_alpha)

        return log_alpha, alpha

    def _state_alpha(self, state, reuse=False):
        self.reset_counter('noisy')
        with tf.variable_scope('temperature', reuse=reuse):
            x = state
            x = self.noisy(x, 1) if self.args['noisy'] else self.dense(x, 1)

            log_alpha = x
            alpha = tf.exp(log_alpha)
        
        return log_alpha, alpha

    def _state_action_alpha(self, state, action, reuse=False):
        self.reset_counter('noisy')
        with tf.variable_scope('temperature', reuse=reuse):
            x = tf.concat([state, action], axis=1)
            x = self.noisy(x, 1) if self.args['noisy'] else self.dense(x, 1)

            log_alpha = x
            alpha = tf.exp(log_alpha)
        
        return log_alpha, alpha