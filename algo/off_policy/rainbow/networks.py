import numpy as np
import tensorflow as tf

from basic_model.basic_nets import Base
from utility.utils import assert_colorize
from algo.off_policy.rainbow import nets_imp

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
        self.atari = args['atari']
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
            net_wrapper = (lambda n_quantiles, batch_size, name, reuse=False: 
                            self._iqn_net(x, 
                                        n_quantiles, 
                                        batch_size, 
                                        self.n_actions,
                                        self._atari_phi_net if else self._cartpole_phi_net,
                                        self._psi_net,
                                        self._cartpole_f_net,
                                        name=name, 
                                        reuse=reuse))
            # online IQN network
            quantiles, quantile_values, Qs = net_wrapper(self.N, self.batch_size, 'main')
            # Qs for online action selection
            _, _, Qs_online = net_wrapper(self.K, 1, 'main', reuse=True)      
            # target IQN network
            _, quantile_values_next_target, Qs_next_target = net_wrapper(self.N_prime, self.batch_size, 'target')
            _, _, Qs_next = net_wrapper(self.K, self.batch_size, 'main', reuse=True)
            
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
            net_fn = self._atari_net if self.atari else self._duel_net
            net_wrapper = lambda name, reuse=False: net_fn(x, self.n_actions, name=name, reuse=reuse)
            Qs = net_wrapper('main')
            Qs_next = net_wrapper(name='main', reuse=True)
            Qs_next_target = net_wrapper(name='target')

            with tf.name_scope('q_values'):
                self.Q = tf.reduce_sum(tf.one_hot(self.action, self.n_actions) * Qs, axis=1, keepdims=True)
                self.best_action = tf.argmax(Qs, axis=1, name='best_action')
                next_action = tf.argmax(Qs_next, axis=1, name='next_action')
                self.Q_next_target = tf.reduce_sum(tf.one_hot(next_action, self.n_actions) 
                                                    * Qs_next_target, axis=1, keepdims=True)

    def _iqn_net(self, state, n_quantiles, batch_size, out_dim, phi_net, psi_net, f_net, name, reuse=None):
        x = state
        quantile_embedding_dim = self.args['quantile_embedding_dim']

        # phi function in the paper
        x_tiled = phi_net(x, n_quantiles, name, reuse)
            
        with tf.name_scope(f'{name}_quantiles'):
            quantile_shape = [n_quantiles * batch_size, 1]
            quantiles = tf.random.uniform(quantile_shape, minval=0, maxval=1)
            quantiles_tiled = tf.tile(quantiles, [1, quantile_embedding_dim])
            # returned quantiles for computing quantile regression loss
            quantiles_reformed = tf.transpose(tf.reshape(quantiles, [n_quantiles, batch_size, 1]), [1, 0, 2])
        
        h_dim = x_tiled.shape.as_list()[1]

        # psi function in the paper
        x_quantiles = psi_net(quantiles_tiled, quantile_embedding_dim, h_dim, name, reuse)

        # Combine outputs of phi and psi
        y = x_tiled * x_quantiles
        # f function in the paper
        quantile_values, q = self.cartpole_f_net(y, out_dim, n_quantiles, batch_size, name, reuse)

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

        h_dim = self.args['h_dim']
        name = f'{name}_net'
        with tf.variable_scope(name, reuse=reuse):
            x = self.dense_norm_activation(x, h_dim, norm=None)
            hv = self.noisy_norm_activation(x, 64, norm=None, name='noisy_relu_v')
            v = self.noisy(hv, out_dim, name='V')
            ha = self.noisy_norm_activation(x, 64, norm=None, name='noisy_relu_a')
            a = self.noisy(ha, out_dim, name='A')

        with tf.variable_scope('Q', reuse=reuse):
            q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        return q

    """ IQN for atari """
    def _atari_phi_net(self, state, out_dim, name, reuse=None):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        assert_colorize(state.shape.as_list()[1:] == [84, 84, 4], 
                f'Input image should be of shape (84, 84, 4), but get {state.shape.as_list()[1:]}')
        x = state

        name = f'{name}_net'
        with tf.variable_scope(f'{name}_phi_net', reuse=reuse):
            x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)      # (19, 19, 32)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)      # (9, 9, 64)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)      # (7, 7, 64)
            x = tf.layers.flatten(x)

        return x
        
    def _psi_net(self, quantiles_tiled, quantile_embedding_dim, h_dim, name, reuse):
        with tf.variable_scope(f'{name}_psi_net', reuse=reuse):
            pi = tf.constant(np.pi)
            x_quantiles = tf.cast(tf.range(quantile_embedding_dim), tf.float32) * pi * quantiles_tiled
            x_quantiles = tf.cos(x_quantiles)
            x_quantiles = tf.layers.dense(x_quantiles, h_dim)

        return x_quantiles
    
    def _atari_f_net(self, x, out_dim, n_quantiles, batch_size, name, reuse):
        with tf.variable_scope(f'{name}_f_net', reuse=reuse):
            x = self.noisy_norm_activation(x, 512, norm=None, name='noisy_relu')
            x = self.noisy(x, out_dim, name='Q')

    """ IQN for cartpole-v0 """
    def _cartpole_phi_net(self, x, n_quantiles, name, reuse):
        h_dim = self.args['h_dim']
        with tf.variable_scope(f'{name}_phi_net', reuse=reuse):
            x = tf.layers.dense(x, h_dim, activation=tf.nn.relu)
            x_tiled = tf.tile(x, [n_quantiles, 1])
        
        return x_tiled

    def _cartpole_f_net(self, x, out_dim, n_quantiles, batch_size, name, reuse):
        with tf.variable_scope(f'{name}_f_net', reuse=reuse):
            x = self.noisy_norm_activation(x, 128, norm=None, name='noisy_relu')
            quantile_values = self.noisy(x, out_dim, name='noisy')
            quantile_values = tf.reshape(quantile_values, (n_quantiles, batch_size, out_dim))
            q = tf.reduce_mean(quantile_values, axis=0)

        return quantile_values, q
