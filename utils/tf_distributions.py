import numpy as np
import tensorflow as tf

class Distribution():
    def neglogp(self, x):
        with tf.name_scope('neg_log_p'):
            return self._neglogp(x)

    def sample(self):
        with tf.name_scope('sample'):
            return self._sample()
        
    def entropy(self):
        with tf.name_scope('entropy'):
            return self._entropy()

    def kl(self, other):
        assert isinstance(other, type(self))
        with tf.name_scope('KL'):
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
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)

    def _sample(self):
        return tf.multinomial(self.logits, 1, output_dtype=tf.int32)

    def _entropy(self):
        probs = self._compute_probs()
        entropy = tf.reduce_sum(-probs * tf.log(probs), axis=-1)

        return entropy

    def _kl(self, other):
        probs = self._compute_probs()
        other_probs = other._compute_probs()
        kl = tf.reduce_sum(probs * tf.log(probs / other_probs), axis=-1)

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
        return (0.5 * tf.log(2 * np.pi) * tf.to_float(tf.shape(x)[-1])
                + tf.reduce_sum(self.logstd, axis=-1)
                + tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1))

    def _sample(self):
        return self.mean + tf.exp(self.logstd) * tf.random.normal(tf.shape(self.mean))

    def _entropy(self):
        return tf.reduce_sum(.5 * np.log(2. * np.pi) + self.logstd + .5, axis=-1)

    def _kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd - .5
                             + .5 * (self.std**2 + (self.mean - other.mean)**2) / other.std**2, axis=-1)