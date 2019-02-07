import tensorflow as tf

def huber_loss(x, delta=1.):
    return tf.where(tf.abs(x) <= delta, 0.5 * tf.square(x), delta * (tf.abs(x) - 0.5 * delta), name='huber_loss')
