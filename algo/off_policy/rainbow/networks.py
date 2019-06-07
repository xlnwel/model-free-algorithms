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
                 action_dim, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.state = data['state']
        self.action = data['action']
        self.next_state = data['next_state']
        self.reward = data['reward']
        self.action_dim = action_dim
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
        x = self._conv_net(self.state, name='main')
        x_next = self._conv_net(self.next_state, name='main', reuse=True)
        x_next_target = self._conv_net(self.next_state, name='target')
        Qs = self._fc_net(x, self.action_dim, name='main')
        Qs_next = self._fc_net(x_next, self.action_dim, name='main', reuse=True)
        Qs_next_target = self._fc_net(x_next_target, self.action_dim, name='target')

        self.Q = tf.reduce_sum(tf.one_hot(self.action, self.action_dim) * Qs, axis=1, keepdims=True)
        self.action = tf.argmax(Qs, axis=1)
        self.next_action = tf.argmax(Qs_next, axis=1)
        self.target_next_Q = tf.reduce_sum(tf.one_hot(self.next_action, self.action_dim) 
                                            * Qs_next_target, axis=1, keepdims=True)

    def _conv_net(self, state, name, reuse=None):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        assert_colorize(state.shape.as_list()[1:] == [84, 84, 4], 
                f'Input image should be of shape (84, 84, 4), but get {state.shape.as_list()[1:]}')
        x = state

        name = f'{name}_conv'
        with tf.variable_scope(name, reuse=reuse):
            x = self.conv_norm_activation(x, 32, 8, 4, padding='same', norm=None)      # (21, 21, 32)
            x = self.conv_norm_activation(x, 64, 4, 2, padding='same', norm=None)      # (11, 11, 64)
            x = self.conv_norm_activation(x, 64, 3, 1, padding='same', norm=None)      # (11, 11, 64)
            x = tf.layers.flatten(x)

        return x

    def _fc_net(self, x, out_dim, name, reuse=None):
        name = f'{name}_fc'
        with tf.variable_scope(name, reuse=reuse):
            x = self.noisy_norm_activation(x, 512, norm=None, name='noisy_relu')
            x = self.noisy(x, out_dim, name='noisy')

        return x
