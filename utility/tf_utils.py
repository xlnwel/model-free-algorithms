import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.keras as tk


# kaiming initializer
def kaiming_initializer(distribution='truncated_normal', seed=None):
    return tk.initializers.VarianceScaling(scale=2., mode='fan_in', distribution=distribution, seed=seed)

# xavier initializer
def xavier_initializer(distribution='truncated_normal', seed=None):
    return tk.initializers.VarianceScaling(scale=1., mode='fan_avg', distribution=distribution, seed=seed)

# batch normalization and relu
def bn_relu(x, training): 
    return tf.nn.relu(tf.layers.batch_normalization(x, training=training))

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

def norm_activation(x, normalization=None, activation=None, training=False):
    if normalization:
        x = (normalization(x, training=training) if
                'batch_normalization' in str(normalization) else
                normalization(x))
    if activation:
        x = activation(x)

    return x

def standard_normalization(x):
    mean, var = tf.nn.moments(x, [0] if len(x.shape.as_list()) == 2 else [0, 1, 2], keepdims=True)
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
    n_step_target = tf.stop_gradient(reward 
                                    + gamma**steps
                                    * (1 - done)
                                    * nth_value, name='n_step_target')

    return n_step_target