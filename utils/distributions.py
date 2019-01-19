import numpy as np
import tensorflow as tf

class Categorical():
    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)

    def sample(self):
        return tf.multinomial(self.logits, 1, output_dtype=tf.int32)

class DiagGaussian():
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd

    def neglogp(self, x):
        return (0.5 * tf.log(2 * np.pi) * tf.to_float(tf.shape(x)[-1])
                + tf.reduce_sum(self.logstd, axis=-1)
                + tf.reduce_sum(tf.square((x - self.mean) / tf.exp(self.logstd)), axis=-1))

    def sample(self):
        return self.mean + tf.exp(self.logstd) * tf.random.normal(tf.shape(self.mean))
