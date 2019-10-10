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
                 next_state,
                 env,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.next_state = next_state
        self.env = env
        self.action_dim = env.action_dim
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None
        self.noisy_sigma = args['noisy_sigma']
        self.has_target_net = args['target']
        self.LOG_STD_MIN = -20.
        self.LOG_STD_MAX = 2.

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
        self.action_det, self.action, self.logpi = self._build_policy(self.state, True, 'main', False)
        if self.has_target_net:
            _, self.next_action, self.next_logpi = self._build_policy(self.next_state, False, 'target', False)
        else:
            _, self.next_action, self.next_logpi = self._build_policy(self.next_state, False, 'main', True)

        self.init_target_op, self.update_target_op = self._target_net_ops()

    def _build_policy(self, state, return_det_action, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            mean, logstd, action_det = self._stochastic_policy_net(state, 
                                                                   self.args['units'], 
                                                                   self.action_dim, 
                                                                   self.norm, 
                                                                   self.noisy_sigma,
                                                                   return_det_action)

            action_distribution = self.env.action_dist_type((mean, logstd))

            orig_action = action_distribution.sample()
            orig_logpi = action_distribution.logp(orig_action)

            # Enforcing action bound
            action, logpi = self._squash_correction(orig_action, orig_logpi)
            
        return action_det, action, logpi

    def _stochastic_policy_net(self, state, units, action_dim, norm, noisy_sigma, return_det_action, name='policy_net'):
        noisy_norm_activation = lambda x, u, norm: self.noisy_norm_activation(x, u, norm=norm, sigma=noisy_sigma)
        x = state
        self.reset_counter('noisy')
        n_noisy = self.args['n_noisy']

        with tf.variable_scope(name):
            for i, u in enumerate(units):
                layer = self.dense_norm_activation if i < len(units) - n_noisy else noisy_norm_activation
                x = layer(x, u, norm=norm)

            # if n_noisy > 0:
            #     mean = self.noisy(x, action_dim)
            #     logstd = self.noisy(x, action_dim)
            # else:
            #     
            mean = self.dense(x, action_dim)
            logstd = self.dense(x, action_dim)

            # constrain logstd to be in range [LOG_STD_MIN, LOG_STD_MAX]
            logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        if return_det_action:
            x_det = state
            with tf.variable_scope(name, reuse=True):
                for u in units:
                    x_det = self.dense_norm_activation(x_det, u, norm=norm)

                mean_det = self.dense(x_det, action_dim)
        else:
            mean_det = None

        return mean, logstd, mean_det

    def _squash_correction(self, action, logpi):
        with tf.name_scope('squash'):
            action_new = tf.tanh(action)
            sub = 2 * tf.reduce_sum(tf.log(2.) + action - tf.nn.softplus(2 * action), axis=1, keepdims=True)
            logpi -= sub

        return action_new, logpi

    def _target_net_ops(self):
        if not self.has_target_net:
            return [], []
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.polyak * v[0] + (1. - self.polyak) * v[1], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op   


class SoftQ(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state,
                 next_state,
                 action,
                 actor,
                 action_dim, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = state
        self.next_state = next_state
        self.action = action
        self.actor_action = actor.action
        self.actor_next_action = actor.next_action
        self.action_dim = action_dim
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None
        self.polyak = args['polyak']
        
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

    """ Implementation """
    def _build_graph(self):
        Q_net = lambda state, action, reuse, name: self._q_net(state, self.args['units'], action, 
                                                        self.norm, reuse, name=name)

        # online network
        with tf.variable_scope('main'):
            self.Q1 = Q_net(self.state, self.action, False, 'Qnet1')
            self.Q2 = Q_net(self.state, self.action, False, 'Qnet2')
            self.Q1_with_actor = Q_net(self.state, self.actor_action, True, 'Qnet1')
            self.Q2_with_actor = Q_net(self.state, self.actor_action, True, 'Qnet2')
            self.Q = tf.minimum(self.Q1, self.Q2, 'Q')
            self.Q_with_actor = tf.minimum(self.Q1_with_actor, self.Q2_with_actor, 'Q_with_actor')

        # target network
        with tf.variable_scope('target'):
            self.next_Q1_with_actor = Q_net(self.next_state, self.actor_next_action, False, 'Qnet1_target')
            self.next_Q2_with_actor = Q_net(self.next_state, self.actor_next_action, False, 'Qnet2_target')
            self.next_Q_with_actor = tf.minimum(self.next_Q1_with_actor, self.next_Q2_with_actor, 'Q_with_actor')

        self.init_target_op, self.update_target_op = self._target_net_ops()

    def _q_net(self, state, units, action, norm, reuse, name='Q_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for i, u in enumerate(units):
                if i < 2:
                    x = tf.concat([x, action], 1)
                x = self.dense_norm_activation(x, u, norm=norm)

            x = self.dense(x, 1, name='Q')

        return x

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.polyak * v[0] + (1. - self.polyak) * v[1], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op   


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
        self.state = state
        self.next_state = next_state
        self.action = policy.action
        self.next_action = policy.next_action
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
            self.next_alpha = self.alpha
        elif self.type == 'state':
            self.log_alpha, self.alpha = self._state_alpha(self.state)
            _, self.next_alpha = self._state_alpha(self.next_state, reuse=True)
        elif self.type == 'state_action':
            self.log_alpha, self.alpha = self._state_action_alpha(self.state, self.action)
            _, self.next_alpha = self._state_action_alpha(self.next_state, self.next_action, reuse=True)

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

