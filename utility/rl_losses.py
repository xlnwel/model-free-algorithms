import tensorflow as tf

def ppo_loss(logpi, old_logpi, advantages, clip_range, entropy, mask=None, n=None):
    with tf.name_scope('ppo_loss'):
        ratio = tf.exp(logpi - old_logpi)
        loss1 = -advantages * ratio
        loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
        if mask is None:
            ppo_loss = tf.reduce_mean(tf.maximum(loss1, loss2), name='ppo_loss')
            entropy = tf.reduce_mean(entropy, name='entropy_loss')
            # debug stats: KL between old and current policy and fraction of data being clipped
            approx_kl = tf.reduce_mean(old_logpi - logpi)
            clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32))
        else:
            ppo_loss = tf.reduce_sum(tf.maximum(loss1, loss2) * mask, name='ppo_loss') / n
            entropy = tf.reduce_sum(entropy * mask, name='entropy_loss') / n
            # debug stats: KL between old and current policy and fraction of data being clipped
            approx_kl = tf.reduce_sum((old_logpi - logpi) * mask) / n
            clipfrac = tf.reduce_sum(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32) * mask) / n
    
    return ppo_loss, entropy, approx_kl, clipfrac

def clipped_value_loss(V, returns, value, clip_range, mask, n):
    with tf.name_scope('clipped_value_loss'):
        V_clipped = value + tf.clip_by_value(V - value, -clip_range, clip_range)
        loss1 = (V - returns)**2
        loss2 = (V_clipped - returns)**2
        if mask is None:
            loss = .5 * tf.reduce_mean(tf.maximum(loss1, loss2))
        else:
            loss = .5 * tf.reduce_sum(tf.maximum(loss1, loss2) * mask) / n
    
    return loss