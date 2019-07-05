import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.keras as tk

from utility.debug_tools import assert_colorize

def kaiming_initializer(distribution='truncated_normal', seed=None):
    """ kaiming initializer """
    return tk.initializers.VarianceScaling(scale=2., mode='fan_in', distribution=distribution, seed=seed)

def xavier_initializer(distribution='truncated_normal', seed=None):
    """ xavier initializer """
    return tk.initializers.VarianceScaling(scale=1., mode='fan_avg', distribution=distribution, seed=seed)

def constant_initializer(val):
    return tk.initializers.Constant(val)

def bn_relu(x, training): 
    """ batch normalization and relu """
    return tf.nn.relu(tf.layers.batch_normalization(x, training=training))

def layer_norm(x, name='LayerNorm', epsilon=1e-5):
    """ Layer normalization """
    with tf.variable_scope(name):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(1, n_dims)), keep_dims=True)
        std = tf.sqrt(var + epsilon)

        x = (x - mean) / std

        shape = (1,)
        gamma = tf.get_variable('gamma', shape=shape, initializer=constant_initializer(1))
        beta = tf.get_variable('beta', shape=shape, initializer=constant_initializer(0))
        x = gamma * x + beta

    return x

def instance_norm(x, name='InstanceNorm', epsilon=1e-5):
    """ Instance normalization """
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        std = tf.sqrt(var + epsilon)

        x = (x - mean) / std

        shape = x.shape.as_list()[-1]
        gamma = tf.get_variable('gamma', shape=shape, initializer=constant_initializer(1))
        beta = tf.get_variable('beta', shape=shape, initializer=constant_initializer(0))

        x = gamma * x + beta

    return x

def norm_activation(x, norm=None, activation=None, training=False, name=None):
    def fn(x):
        if norm:
            x = (norm(x, training=training) if
                    'batch_normalization' in str(norm) else
                    norm(x))
        if activation:
            x = activation(x)

    if name:
        with tf.variable_scope(name):
            fn(x)
    else:
        fn(x)

    return x

def standard_normalization(x):
    with tf.variable_scope('Normalization'):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(n_dims-1)), keep_dims=True)
        std = tf.sqrt(var)

        x = (x - mean) / std
    
    return x

def logsumexp(value, axis=None, keepdims=False):
    if axis is not None:
        max_value = tf.reduce_max(value, axis=axis, keepdims=True)
        value0 = value - max_value    # for numerical stability
        if keepdims is False:
            max_value = tf.squeeze(max_value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value0),
                                                axis=axis, keepdims=keepdims))
    else:
        max_value = tf.reduce_max(value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value - max_value)))

def get_tensor(sess, name=None, op_name=None):
    if name is None and op_name is None:
        raise ValueError
    elif name:
        return sess.graph.get_tensor_by_name(name)
    else:
        return sess.graph.get_tensor_by_name(op_name + ':0')

def n_step_target(reward, done, nth_value, gamma, steps=1):
    with tf.variable_scope('n_step_target'):
        n_step_target = tf.stop_gradient(reward 
                                        + gamma**steps
                                        * (1 - done)
                                        * nth_value, name='n_step_target')

    return n_step_target

def stats_summary(data, name, max=True, min=True, mean=True, hist=True):
    if max:
        tf.summary.scalar(f'{name}_max_', tf.reduce_max(data))
    if min:
        tf.summary.scalar(f'{name}_min_', tf.reduce_min(data))
    if mean:
        tf.summary.scalar(f'{name}_avg_', tf.reduce_mean(data))
    if hist:
        tf.summary.histogram(f'{name}_', data)

def get_vars(scope, graph=tf.get_default_graph()):
    return [x for x in graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)]

def count_vars(scope, graph=tf.get_default_graph()):
    v = get_vars(scope, graph=graph)
    return sum([np.prod(var.shape.as_list()) for var in v])

def padding(x, height, width, mode='constant', name=None):
    assert_colorize(mode.lower() == 'constant' or mode.lower() == 'reflect' or mode.lower() == 'symmetric', 
        f'Padding should be "constant", "reflect", or "symmetric", but got {mode}.')
    return tf.pad(x, [[0, 0], [height, height], [width, width], [0, 0]], mode, name=name)
