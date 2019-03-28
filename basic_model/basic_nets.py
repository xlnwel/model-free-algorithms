import tensorflow as tf
from tensorflow.contrib import layers
from basic_model.model import Module


class Base(Module):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.variable_scope = scope_prefix + '/' + name
        
        super().__init__(name, args, graph, log_tensorboard=log_tensorboard, log_params=log_params)

    def _Q_net(self, state, units, action, action_space, reuse, name='Q_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for i, u in enumerate(units):
                if i == 1:
                    x = tf.concat([x, action], 1)
                x = self.dense_norm_activation(x, u)

            x = self.dense(x, 1, name='Q')

        return x

    def _V_net(self, state, units, name='V_net'):
        x = state
        with tf.variable_scope(name):
            for u in units:
                x = self.dense_norm_activation(x, u, normalization=None)
            x = self.dense(x, 1, name='V')

        return x