import tensorflow as tf

def huber_loss(x, y=None, delta=1.):
    if y != None:   # if y is passed, take x-y as error, otherwise, take x as error
        x = x - y
    return tf.where(tf.abs(x) <= delta, 0.5 * tf.square(x), delta * (tf.abs(x) - 0.5 * delta), name='huber_loss')
