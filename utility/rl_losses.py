import tensorflow as tf


def ppo_loss(logpi, old_logpi, advantages, clip_range, entropy, mask=None, n=None):
    assert (mask is None) == (n is None), f'Both/Neither mask and n should be None, but get \nmask:{mask}\nn:{n}'
    if mask is None:
        reduce_mean = lambda x, name: tf.reduce_mean(x, name=name)
    else:
        reduce_mean = lambda x, name: tf.divide(tf.reduce_sum(x), n, name=name)
    with tf.name_scope('ppo_loss'):
        ratio = tf.exp(logpi - old_logpi)
        loss1 = -advantages * ratio
        loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
        if mask is None:
            mask = 1.
        ppo_loss = reduce_mean(tf.maximum(loss1, loss2) * mask, name='ppo_loss')
        entropy = reduce_mean(entropy * mask, name='entropy_loss')
        # debug stats: KL between old and current policy and fraction of data being clipped
        approx_kl = .5 * reduce_mean((old_logpi - logpi)**2 * mask, name='approx_kl')
        clip_frac = reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32) * mask, 'clip_frac')
    
    return ppo_loss, entropy, approx_kl, clip_frac

def clipped_value_loss(value, traj_ret, old_value, clip_range, mask=None, n=None):
    assert (mask is None) == (n is None), f'Both/Neither mask and n should be None, but get \nmask:{mask}\nn:{n}'
    if mask is None:
        reduce_mean = lambda x, name: tf.reduce_mean(x, name=name)
    else:
        reduce_mean = lambda x, name: tf.divide(tf.reduce_sum(x), n, name=name)
    with tf.name_scope('clipped_value_loss'):
        V_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
        loss1 = (value - traj_ret)**2
        loss2 = (V_clipped - traj_ret)**2
        if mask is None:
            mask = 1.
        loss = reduce_mean(tf.maximum(loss1, loss2) * mask, name='value_loss')
    
    return loss