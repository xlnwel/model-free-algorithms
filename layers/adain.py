import tensorflow as tf


class AdaIN:
    """Adaptive Instance Normalization
    """
    def __init__(self, name='AdaIN'):
        self.name = name

    def __call__(self, content_inputs, style_inputs):
        with tf.name_scope(self.name):
            content_inputs = tf.convert_to_tensor(content_inputs)
            style_inputs = tf.convert_to_tensor(style_inputs)

            content_mean, content_variance = tf.nn.moments(content_inputs, [1, 2], keep_dims=True)
            style_mean, style_variance = tf.nn.moments(style_inputs, [1, 2], keep_dims=True)

            outputs = style_variance * (content_inputs - content_mean) / content_variance + style_mean

        return outputs

def adaptive_instance_norm(content_inputs, style_inputs, name='AdaIN'):
    adain = AdaIN(name)
    return adain(content_inputs, style_inputs)