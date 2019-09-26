import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.keras as tk

from utility.debug_tools import assert_colorize
from layers.adain import adaptive_instance_norm

def kaiming_initializer(distribution='truncated_normal', seed=None):
    """ kaiming initializer """
    return tk.initializers.VarianceScaling(scale=2., mode='fan_in', distribution=distribution, seed=seed)

def xavier_initializer(distribution='truncated_normal', seed=None):
    """ xavier initializer """
    return tk.initializers.VarianceScaling(scale=1., mode='fan_avg', distribution=distribution, seed=seed)

def constant_initializer(val):
    return tk.initializers.Constant(val)

def layer_norm(x, name='LayerNorm', epsilon=1e-5):
    """ Layer normalization """
    with tf.variable_scope(name):
        n_dims = len(x.shape.as_list())
        mean, var = tf.nn.moments(x, list(range(1, n_dims)), keep_dims=True)
        std = tf.sqrt(var + epsilon)

        x = (x - mean) / std

        shape = x.shape[1:]
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

        shape = x.shape[-1:]
        gamma = tf.get_variable('gamma', shape=shape, initializer=constant_initializer(1))
        beta = tf.get_variable('beta', shape=shape, initializer=constant_initializer(0))

        x = gamma * x + beta

    return x

def upsample(x):
    h, w = x.get_shape().as_list()[1:-1]
    x = tf.image.resize_nearest_neighbor(x, [2 * h, 2 * w])
    return x

def norm_activation(x, norm=None, activation=None, training=False, name=None):
    def fn():
        y = x
        if norm:
            y = (norm(y, training=training) if
                    'batch_norm' in str(norm) else
                    norm(y))
        if activation:
            y = activation(y)
        return y

    x = wrap_layer(name, fn)

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

def square_sum(x):
    return 2 * tf.nn.l2_loss(x)
    
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

def stats_summary(name, data, mean=True, std=False, max=False, min=False, hist=False):
    if mean:
        tf.summary.scalar(f'{name}_mean_', tf.reduce_mean(data))
    if std:
        tf.summary.scalar(f'{name}_std_', tf.reduce_mean(data))
    if max:
        tf.summary.scalar(f'{name}_max_', tf.reduce_max(data))
    if min:
        tf.summary.scalar(f'{name}_min_', tf.reduce_min(data))
    if hist:
        tf.summary.histogram(f'{name}_', data)

def get_vars(scope, graph=tf.get_default_graph()):
    return [x for x in graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)]

def count_vars(scope, graph=tf.get_default_graph()):
    v = get_vars(scope, graph=graph)
    return sum([np.prod(var.shape.as_list()) for var in v])

def padding(x, kernel_size, strides, mode='constant', name=None):
    """ This function pads x so that a convolution with the same args downsamples x by a factor of strides.
    It achieves it using the following equation:
    W // S = (W - k_w + 2P) / S + 1
    """
    assert_colorize(mode.lower() == 'constant' or mode.lower() == 'reflect' or mode.lower() == 'symmetric', 
        f'Padding should be "constant", "reflect", or "symmetric", but got {mode}.')
    H, W = x.shape.as_list()[1:3]
    if isinstance(kernel_size, list) and len(kernel_size) == 2:
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    p_h1 = int(((H / strides - 1) * strides - H + k_h) // strides)
    p_h2 = int(((H / strides - 1) * strides - H + k_h) - p_h1)
    p_w1 = int(((W / strides - 1) * strides - W + k_w) // strides)
    p_w2 = int(((W / strides - 1) * strides - W + k_w) -p_w1)
    return tf.pad(x, [[0, 0], [p_h1, p_h2], [p_w1, p_w2], [0, 0]], mode, name=name)

def wrap_layer(name, layer_imp):
    if name:
        with tf.variable_scope(name):
            x = layer_imp()
    else:
        x = layer_imp()

    return x

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])    # [N, M]

    # [1, M]
    u_var = tf.get_variable('u', [1, w_shape[-1]], 
                            initializer=tf.truncated_normal_initializer(), 
                            trainable=False)
    u = u_var
    # power iteration
    for _ in range(iteration):
        v = tf.nn.l2_normalize(tf.matmul(u, w, transpose_b=True))           # [1, N]
        u = tf.nn.l2_normalize(tf.matmul(v, w))                             # [1, M]

    sigma = tf.squeeze(tf.matmul(tf.matmul(v, w), u, transpose_b=True))     # scalar
    w = w / sigma

    with tf.control_dependencies([u_var.assign(u)]):                        # we reuse the value of u
        w = tf.reshape(w, w_shape)

    return w

def positional_encoding(indices, max_idx, dim, name='positional_encoding'):
    with tf.name_scope(name):
        # exp(-2i / d_model * log(10000))
        vals = np.array([pos * np.exp(- np.arange(0, dim, 2) / dim * np.log(10000)) for pos in range(max_idx)])
        
        params = np.zeros((max_idx, dim))
        params[:, 0::2] = np.sin(vals)    # 2i
        params[:, 1::2] = np.cos(vals)    # 2i + 1
        params = tf.convert_to_tensor(params, tf.float32)

        v = tf.nn.embedding_lookup(params, indices)

    return v

def get_norm(name):
    if name == 'instance':
        return tc.layers.instance_norm
    elif name == 'layer':
        return tc.layers.layer_norm
    elif name == 'batch':
        return tf.layers.batch_normalization
    elif name == 'adain':
        return adaptive_instance_norm
    elif name is None or name == 'None':
        return None
    else:
        raise NotImplementedError
