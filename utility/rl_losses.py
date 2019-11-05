import tensorflow as tf


def ppo_loss(logpi, old_logpi, advantages, clip_range, entropy, mask=None, n=None):
    assert (mask is None) == (n is None), f'Both/Neither mask and n should be None, but get \nmask:{mask}\nn:{n}'
    def reduce_mean(x, name):
        with tf.name_scope(name):        
            return tf.reduce_mean(x) if mask is None else tf.reduce_sum(x) / n

    with tf.name_scope('ppo_loss'):
        ratio = tf.exp(logpi - old_logpi)
        loss1 = -advantages * ratio
        loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
        
        m = 1. if mask is None else mask
        
        pg_loss = reduce_mean(tf.maximum(loss1, loss2) * m, name='ppo_loss')
        entropy = tf.reduce_mean(entropy, name='entropy_loss')
        # debug stats: KL between old and current policy and fraction of data being clipped
        approx_kl = .5 * reduce_mean((old_logpi - logpi)**2 * m, name='approx_kl')
        clip_frac = reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32) * m, 'clip_frac')
    
    return pg_loss, entropy, approx_kl, clip_frac

def clipped_value_loss(value, traj_ret, old_value, clip_range, mask=None, n=None):
    assert (mask is None) == (n is None), f'Both/Neither mask and n should be None, but get \nmask:{mask}\nn:{n}'
    def reduce_mean(x, name):
        with tf.name_scope(name):        
            return tf.reduce_mean(x) if mask is None else tf.reduce_sum(x) / n

    with tf.name_scope('clipped_value_loss'):
        V_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
        loss1 = (value - traj_ret)**2
        loss2 = (V_clipped - traj_ret)**2
        
        m = 1. if mask is None else mask
        
        loss = reduce_mean(tf.maximum(loss1, loss2) * m, name='value_loss')
    
    return loss