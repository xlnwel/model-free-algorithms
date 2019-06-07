import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.keras as tk


# kaiming initializer
def kaiming_initializer(distribution='truncated_normal', seed=None):
    return tk.initializers.VarianceScaling(scale=2., mode='fan_in', distribution=distribution, seed=seed)

# xavier initializer
def xavier_initializer(distribution='truncated_normal', seed=None):
    return tk.initializers.VarianceScaling(scale=1., mode='fan_avg', distribution=distribution, seed=seed)

def constant_initializer(val):
    return tk.initializers.Constant(val)

# batch normalization and relu
def bn_relu(x, training): 
    return tf.nn.relu(tf.layers.batch_normalization(x, training=training))

def layer_norm(x, name='LayerNorm'):
    with tf.variable_scope(name):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(1, n_dims)), keep_dims=True)
        std = tf.sqrt(var)

        x = (x - mean) / std

        shape = x.shape.as_list[1:]
        gamma = tf.get_variable('gamma', shape=shape, initializer=constant_initializer(1))
        beta = tf.get_variable('gamma', shape=shape, initializer=constant_initializer(0))
        x = gamma * x + beta

    return x

# layer normalization and relu
def ln_relu(x):
    return tf.nn.relu(tc.layers.layer_norm(x))

def bn_activation(x, training, activation=None, return_layer_obj=False):
    x = tc.layers.batch_normalization(x, training=training)

    if activation:
        x = activation(x)

    return x

def ln_activation(x, activation=None, return_layer_obj=False):
    x = tc.layers.layer_norm(x)

    if activation:
        x = activation(x)

    return x

def norm_activation(x, norm=None, activation=None, training=False):
    if norm:
        x = (norm(x, training=training) if
                'batch_normalization' in str(norm) else
                norm(x))
    if activation:
        x = activation(x)

    return x

def standard_normalization(x):
    with tf.variable_scope('Normalization'):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(n_dims-1)), keep_dims=True)
        std = tf.sqrt(var)

        x = (x - mean) / std
    
    return x

def range_normalization(images, normalizing=True):
    if normalizing:
        processed_images = tf.cast(images, tf.float32) / 128 - 1
    else:
        processed_images = tf.cast((tf.clip_by_value(images, -1, 1) + 1) * 128, tf.uint8)

    return processed_images

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
