import numpy as np
import tensorflow as tf

class Distribution():
    def neglogp(self, x):
        with tf.name_scope('neg_log_p'):
            return self._neglogp(x)

    def sample(self):
        with tf.name_scope('sample'):
            return self._sample()

    def _neglogp(self, x):
        raise NotImplementedError

    def _sample(self):
        raise NotImplementedError

class Categorical(Distribution):
    def __init__(self, logits):
        self.logits = logits

    def _neglogp(self, x):
        x = tf.squeeze(x)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)

    def _sample(self):
        return tf.multinomial(self.logits, 1, output_dtype=tf.int32)


class DiagGaussian(Distribution):
    def __init__(self, params):
        self.mean, self.logstd = params

    def _neglogp(self, x):
        return (0.5 * tf.log(2 * np.pi) * tf.to_float(tf.shape(x)[-1])
                + tf.reduce_sum(self.logstd, axis=-1)
                + tf.reduce_sum(tf.square((x - self.mean) / tf.exp(self.logstd)), axis=-1))

    def _sample(self):
        return self.mean + tf.exp(self.logstd) * tf.random.normal(tf.shape(self.mean))
