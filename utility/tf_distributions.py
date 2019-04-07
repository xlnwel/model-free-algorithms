import numpy as np
import tensorflow as tf

EPSILON = 1e-8


def tf_scope(func):
    def name_scope(*args):
        with tf.name_scope(func.__name__):
            return func(*args)
    return name_scope

class Distribution():
    @tf_scope
    def logp(self, x):
        return -self._neglogp(x)

    @tf_scope
    def neglogp(self, x):
        return self._neglogp(x)

    @tf_scope
    def sample(self):
        return self._sample()
        
    @tf_scope
    def entropy(self):
        return self._entropy()

    @tf_scope
    def kl(self, other):
        assert isinstance(other, type(self))
        return self._kl(other)

    def _neglogp(self, x):
        raise NotImplementedError

    def _sample(self):
        raise NotImplementedError

    def _entropy(self):
        raise NotImplementedError

    def _kl(self, other):
        raise NotImplementedError


class Categorical(Distribution):
    def __init__(self, logits):
        self.logits = logits

    def _neglogp(self, x):
        x = tf.reshape(x, [-1])
        return tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits), [-1, 1])

    def _sample(self):
        return tf.random.categorical(self.logits, 1, dtype=tf.int32)

    def _entropy(self):
        probs = self._compute_probs()
        entropy = tf.reduce_sum(-probs * tf.log(probs), axis=-1)

        return entropy

    def _kl(self, other):
        probs = self._compute_probs()
        other_probs = other._compute_probs()
        kl = tf.reduce_sum(probs * (tf.log(probs) - tf.log(other_probs)), axis=-1)

        return kl

    def _compute_probs(self):
        logits = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_logits = tf.exp(logits)
        sum_exp_logits = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / sum_exp_logits

        return probs

class DiagGaussian(Distribution):
    def __init__(self, params):
        self.mean, self.logstd = params
        self.std = tf.exp(self.logstd)

    def _neglogp(self, x):
        return .5 * tf.reduce_sum(np.log(2. * np.pi)
                                  + 2 * self.logstd
                                  + ((x - self.mean) / (self.std + EPSILON))**2, 
                                  axis=-1, keepdims=True)

    def _sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def _entropy(self):
        return tf.reduce_sum(.5 * np.log(2. * np.pi) + self.logstd + .5, axis=-1)

    def _kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd - .5
                             + .5 * (self.std**2 + (self.mean - other.mean)**2) / (other.std + EPSILON)**2, axis=-1)
