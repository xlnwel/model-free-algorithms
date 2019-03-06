import tensorflow as tf

from basic_model.model import Module


class Base(Module):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 state, 
                 reuse=False, 
                 scope_prefix='',
                 log_tensorboard=True,
                 log_params=False):
        self.state = state
        self._variable_scope = scope_prefix + '/' + name
        
        super().__init__(name, args, graph, reuse=reuse, log_tensorboard=log_tensorboard, log_params=log_params)

    def _deterministic_policy_net(self, state, action_dim, noisy_sigma, reuse, name='policy_net'):
        with tf.variable_scope(name, reuse=reuse):
            x = self._dense(state, 512)
            x = self._dense_resnet_norm_activation(x, 512)
            x = self._noisy_norm_activation(x, 256, sigma=noisy_sigma)
            x = self._noisy(x, action_dim, sigma=noisy_sigma)
            x = tf.tanh(x, name='action')

        return x

    # TODO: try using noisy layer here
    def _stochastic_policy_net(self, state, action_dim, reuse, discrete=False, name='policy_net'):
        with tf.variable_scope(name, reuse=reuse):
            x = self._dense_norm_activation(state, 512, activation=None)
            x = self._dense_resnet_norm_activation(x, 512)
            x = self._dense_norm_activation(x, 256)
            output_name = ('action_logits' if discrete else 'action_mean')
            x = self._dense(x, action_dim, name=output_name)

        if discrete:
            return x
        else:
            logstd = tf.get_variable('action_logstd', [action_dim], tf.float32)
            return x, logstd

    def _Q_net(self, state, action, action_dim, reuse, name='Q_net'):
        self._reset_counter('dense_resnet')

        with tf.variable_scope(name, reuse=reuse):
            x = self._dense_norm_activation(state, 512 - action_dim, activation=None)
            x = tf.concat([x, action], 1)
            x = self._dense_resnet_norm_activation(x, 512)
            x = self._dense_norm_activation(x, 256)
            x = self._dense(x, 1, name='Q')

        return x

    def _V_net(self, state, reuse, name='V_net'):
        with tf.variable_scope(name, reuse=reuse):
            x = self._dense_norm_activation(state, 512, activation=None)
            x = self._dense_resnet_norm_activation(x, 512)
            x = self._dense_norm_activation(x, 256)
            x = self._dense(x, 1, name='V')

        return x