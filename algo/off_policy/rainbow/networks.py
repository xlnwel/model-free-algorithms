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
        Qs = self._net(x, self.n_actions, name='main')
        Qs_next = self._net(x, self.n_actions, name='main', reuse=True)
        Qs_next_target = self._net(x, self.n_actions, name='target')

        self.Q = tf.reduce_sum(tf.one_hot(self.action, self.n_actions) * Qs, axis=1, keepdims=True)
        self.action = tf.argmax(Qs, axis=1, name='action')
        self.next_action = tf.argmax(Qs_next, axis=1, name='next_action')
        self.target_next_Q = tf.reduce_sum(tf.one_hot(self.next_action, self.n_actions) 
                                            * Qs_next_target, axis=1, keepdims=True)

    def _net(self, state, out_dim, name, reuse=None):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        assert_colorize(state.shape.as_list()[1:] == [84, 84, 4], 
                f'Input image should be of shape (84, 84, 4), but get {state.shape.as_list()[1:]}')
        x = state

        name = f'{name}_net'
        with tf.variable_scope(name, reuse=reuse):
            x = self.conv_norm_activation(x, 32, 8, 4, padding='same', norm=None)      # (21, 21, 32)
            x = self.conv_norm_activation(x, 64, 4, 2, padding='same', norm=None)      # (11, 11, 64)
            x = self.conv_norm_activation(x, 64, 3, 1, padding='same', norm=None)      # (11, 11, 64)
            x = tf.layers.flatten(x)
            x = self.noisy_norm_activation(x, 512, norm=None, name='noisy_relu')
            x = self.noisy(x, out_dim, name='noisy')

        return x
