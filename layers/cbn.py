import tensorflow as tf


class ConditionalBatchNorm:
    """Conditional BatchNorm.
    For each class, it has a specific gamma and beta as normalization variable.
    """
    def __init__(self, n_classes, name='conditional_batch_norm', decay_rate=0.999):
        self.name = name
        self.n_classes = n_classes
        self.decay_rate = decay_rate

    def __call__(self, inputs, labels, is_training=True):
        # denote number of classes as N, number of features(channels) as F, length of labels as L 
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]  # F
        shape = tf.TensorShape([self.n_classes]).concatenate(params_shape)      # shape [N, F]
        moving_shape = tf.TensorShape([1, 1, 1]).concatenate(params_shape)      # shape [1, 1, 1, F]

        with tf.variable_scope(self.name):
            # [N, F]
            self.gamma = tf.get_variable(
                'gamma', shape,
                initializer=tf.ones_initializer())
            # [N, F]
            self.beta = tf.get_variable(
                'beta', shape,
                initializer=tf.zeros_initializer())
            # [1, 1, 1, F]
            self.moving_mean = tf.get_variable('mean', moving_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
            # [1, 1, 1, F]
            self.moving_var = tf.get_variable('var', moving_shape,
                                initializer=tf.ones_initializer(),
                                trainable=False)

            beta = tf.gather(self.beta, labels)   # [L, F]
            beta = tf.expand_dims(tf.expand_dims(beta, 1), 1) # [L, 1, 1, F]
            gamma = tf.gather(self.gamma, labels) # [L, F]
            gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1) # [L, 1, 1, F]
            decay = self.decay_rate
            variance_epsilon = 1e-5
            if is_training:
                mean, variance = tf.nn.moments(inputs, [0, 1, 2], keep_dims=True)
                update_mean = tf.assign(self.moving_mean, self.moving_mean * decay + mean * (1 - decay))
                update_var = tf.assign(self.moving_var, self.moving_var * decay + variance * (1 - decay))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
                outputs = tf.nn.batch_normalization(
                    inputs, mean, variance, beta, gamma, variance_epsilon)
            else:
                outputs = tf.nn.batch_normalization(
                    inputs, self.moving_mean, self.moving_var, beta, gamma, variance_epsilon)
            outputs.set_shape(inputs_shape)
            return outputs

def conditional_batch_norm(inputs, labels, n_classes, 
                            decay_rate=0.999, is_training=True, 
                            name='conditional_batch_norm'):
    cbn = ConditionalBatchNorm(n_classes, name=name, decay_rate=decay_rate)
    return cbn(inputs, labels, is_training=is_training)
