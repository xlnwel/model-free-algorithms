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
        self.polyak = args['polyak']
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
        x_next = self._conv_net(self.next_state, name='main')
        x_next_target = self._conv_net(self.next_state, name='target')
        self.Q = self._fc_net(x, self.action_dim, name='main')
        self.Q_next = self._fc_net(x_next, self.action_dim, name='main')
        self.Q_next_target = self._fc_net(x_next_target, self.action_dim, name='target')

        # self.Q = tf.reduce_sum(tf.one_hot(self.action, self.action_dim) * self.Q_dist, axis=1)
        self.action = tf.argmax(self.Q, axis=1)
        self.next_action = tf.argmax(self.Q_next, axis=1)
        # self.target_next_Q = tf.reduce_sum(tf.one_hot(self.next_action, self.action_dim) * self.target_Q_dist, axis=1)

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

    def _conv_net(self, state, name):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        assert_colorize(state.shape.as_list()[1:] == [84, 84, 1], f'Input image should be of shape (84, 84, 1), but get {state.shape.as_list()[1:]}')
        x = state

        name = f'{name}_conv'
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = self.conv_norm_activation(x, 32, 8, 4, padding='same', norm=None)      # (21, 21, 32)
            x = self.conv_norm_activation(x, 64, 4, 2, padding='same', norm=None)      # (11, 11, 64)
            x = self.conv_norm_activation(x, 64, 3, 1, padding='same', norm=None)      # (11, 11, 64)
            x = tf.layers.flatten(x)

        return x

    def _fc_net(self, x, out_dim, name):
        name = f'{name}_fc'
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = self.noisy_norm_activation(x, 512, norm=None)
            x = self.noisy(x, out_dim)

        return x

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.polyak * v[0] + (1. - self.polyak) * v[1], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op
