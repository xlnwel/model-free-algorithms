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
        self.variable_scope = scope_prefix + '/' + name
        
        super().__init__(name, args, graph, reuse=reuse, log_tensorboard=log_tensorboard, log_params=log_params)

    def _deterministic_policy_net(self, state, units, action_space, noisy_sigma, reuse, name='policy_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for u in units:
                x = self.noisy_norm_activation(x, u)
            x = self.noisy(x, action_space, sigma=noisy_sigma)
            x = tf.tanh(x, name='action')

        return x

    # TODO: try using noisy layer here
    def _stochastic_policy_net(self, state, units, action_space, reuse, discrete=False, simple_logstd=True, name='policy_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for u in units:
                x = self.dense_norm_activation(x, u)
            output_name = ('action_logits' if discrete else 'action_mean')
            y = self.dense(x, action_space, name=output_name)

            if discrete:
                return y
            else:
                if simple_logstd:
                    logstd = tf.get_variable('action_logstd', [action_space], tf.float32)
                else:
                    logstd = self.dense_norm_activation(x, action_space, name='action_logstd', 
                                                        normalization=None, activation=tf.tanh)
                return x, logstd

    def _Q_net(self, state, units, action, action_space, reuse, name='Q_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for i, u in enumerate(units):
                if i == 1:
                    x = tf.concat([x, action], 1)
                x = self.dense_norm_activation(x, u)

            x = self.dense(x, 1, name='Q')

        return x

    def _V_net(self, state, units, reuse, name='V_net'):
        x = state
        with tf.variable_scope(name, reuse=reuse):
            for u in units:
                x = self.dense_norm_activation(x, u)
            x = self.dense(x, 1, name='V')

        return x