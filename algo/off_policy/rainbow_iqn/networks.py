import numpy as np
import tensorflow as tf

from basic_model.basic_nets import Base
from utility.utils import assert_colorize

class Networks(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 data,
                 n_actions, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = data['state']
        self.action = data['action']
        self.next_state = data['next_state']
        self.reward = data['reward']
        self.n_actions = n_actions
        self.batch_size = args['batch_size']
        self.N = args['N']                          # N in paper, num of quantiles for online quantile network
        self.N_prime = args['N_prime']              # N' in paper, num of quantiles for target quantile network
        self.K = args['K']                          # K in paper, num of quantiles for action selection
        self.delta = args['delta']                  # kappa in paper, used in huber loss

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
        x = self.state
        if self.args['iqn']:
            net_fn = (lambda n_quantiles, batch_size, name, reuse=False: 
                            self._iqn_net(x, 
                                        n_quantiles, 
                                        batch_size, 
                                        self.n_actions,
                                        self._psi_net,
                                        self._phi_net,
                                        self._f_net,
                                        name=name, 
                                        reuse=reuse))
            # online IQN network
            quantiles, quantile_values, Qs = net_fn(self.N, self.batch_size, 'main')
            # Qs for online action selection
            _, _, Qs_online = net_fn(self.K, 1, 'main', reuse=True)      
            # target IQN network
            _, quantile_values_next_target, Qs_next_target = net_fn(self.N_prime, self.batch_size, 'target')
            _, _, Qs_next = net_fn(self.K, self.batch_size, 'main', reuse=True)
            
            self.quantiles = quantiles                                              # [B, N, 1]
            self.best_action = self._iqn_action(Qs_online, 'best_action')           # [1]
            next_action = self._iqn_action(Qs_next, 'next_action')                  # [B]

            # quantile_values for regression loss
            # Q for priority required by PER
            self.quantile_values, self.Q = self._iqn_values(self.action, self.N, quantile_values, Qs)
            self.quantile_values_next_target, self.Q_next_target = self._iqn_values(next_action, 
                                                                                    self.N_prime,
                                                                                    quantile_values_next_target, 
                                                                                    Qs_next_target)
        else: 
            net_fn = lambda name, reuse=False: self._duel_net(x, self.n_actions, name=name, reuse=reuse)
            Qs = net_fn('main')
            Qs_next = net_fn(name='main', reuse=True)
            Qs_next_target = net_fn(name='target')

            with tf.name_scope('q_values'):
                self.Q = tf.reduce_sum(tf.one_hot(self.action, self.n_actions) * Qs, axis=1, keepdims=True)
                self.best_action = tf.argmax(Qs, axis=1, name='best_action')
                next_action = tf.argmax(Qs_next, axis=1, name='next_action')
                self.Q_next_target = tf.reduce_sum(tf.one_hot(next_action, self.n_actions) 
                                                    * Qs_next_target, axis=1, keepdims=True)

    def _iqn_net(self, state, n_quantiles, batch_size, out_dim, psi_net, phi_net, f_net, name, reuse=None):
        x = state
        quantile_embedding_dim = self.args['quantile_embedding_dim']

        # psi function in the paper
        x_tiled = psi_net(x, n_quantiles, name, reuse)
            
        with tf.name_scope(f'{name}_quantiles'):
            quantile_shape = [n_quantiles * batch_size, 1]
            quantiles = tf.random.uniform(quantile_shape, minval=0, maxval=1)
            quantiles_tiled = tf.tile(quantiles, [1, quantile_embedding_dim])
            # returned quantiles for computing quantile regression loss
            quantiles_reformed = tf.transpose(tf.reshape(quantiles, [n_quantiles, batch_size, 1]), [1, 0, 2])
        
        h_dim = x_tiled.shape.as_list()[1]

        # phi function in the paper
        x_quantiles = phi_net(quantiles_tiled, quantile_embedding_dim, h_dim, name, reuse)

        # Combine outputs of psi and phi
        y = x_tiled * x_quantiles
        # f function in the paper
        quantile_values, q = f_net(y, out_dim, n_quantiles, batch_size, name, reuse)

        return quantiles_reformed, quantile_values, q

    def _iqn_action(self, Qs, name):
        with tf.name_scope(name):
            action = tf.argmax(Qs, axis=1, name='best_action')
        
        return action
        
    def _iqn_values(self, action, n_quantiles, quantile_values, Qs):
        with tf.name_scope('quantiles'):
            action_tiled = tf.reshape(tf.tile(action, [n_quantiles]), 
                                        [n_quantiles, -1])
            quantile_values = tf.reduce_sum(tf.one_hot(action_tiled, self.n_actions)
                                            * quantile_values, axis=2, keepdims=True)
        with tf.name_scope('q_values'):
            Q = tf.reduce_sum(tf.one_hot(action, self.n_actions)
                              * Qs, axis=1, keepdims=True)

        return quantile_values, Q

    def _duel_net(self, state, out_dim, name, reuse=None):
        x = state

        name = f'{name}_net'
        with tf.variable_scope(name, reuse=reuse):
            for u in self.args['psi_units']:
                x = self.dense_norm_activation(x, u, norm=None)
            for u in self.args['f_units']:
                hv = self.noisy_norm_activation(x, 64, norm=None, name='noisy_relu_v')
            v = self.noisy(hv, out_dim, name='V')
            for u in self.args['f_units']:
                ha = self.noisy_norm_activation(x, 64, norm=None, name='noisy_relu_a')
            a = self.noisy(ha, out_dim, name='A')

        with tf.variable_scope('Q', reuse=reuse):
            q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        return q

    """ IQN for cartpole-v0 """
    def _psi_net(self, x, n_quantiles, name, reuse):
        with tf.variable_scope(f'{name}_psi_net', reuse=reuse):
            for u in self.args['psi_units']:
                x = tf.layers.dense(x, u, activation=tf.nn.relu)
            x_tiled = tf.tile(x, [n_quantiles, 1])
        
        return x_tiled

    def _f_net(self, x, out_dim, n_quantiles, batch_size, name, reuse):
        with tf.variable_scope(f'{name}_f_net', reuse=reuse):
            for u in self.args['f_units']:
                x = self.noisy_norm_activation(x, u, norm=None, name='noisy_relu')
            quantile_values = self.noisy(x, out_dim, name='noisy')
            quantile_values = tf.reshape(quantile_values, (n_quantiles, batch_size, out_dim))
            q = tf.reduce_mean(quantile_values, axis=0)

        return quantile_values, q

    def _phi_net(self, quantiles_tiled, quantile_embedding_dim, h_dim, name, reuse):
        with tf.variable_scope(f'{name}_phi_net', reuse=reuse):
            pi = tf.constant(np.pi)
            x_quantiles = tf.cast(tf.range(quantile_embedding_dim), tf.float32) * pi * quantiles_tiled
            x_quantiles = tf.cos(x_quantiles)
            x_quantiles = tf.layers.dense(x_quantiles, h_dim)

        return x_quantiles