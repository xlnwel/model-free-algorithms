import numpy as np
import tensorflow as tf


h_dim = 256

def cartpole_phi_net(x, n_quantiles, name, reuse):

    with tf.variable_scope(f'{name}_phi_net', reuse=reuse):
        x = tf.layers.dense(x, h_dim, activation=tf.nn.relu)
        x_tiled = tf.tile(x, [n_quantiles, 1])
    
    return x_tiled

def cartpole_psi_net(quantiles_tiled, quantile_embedding_dim, name, reuse):
    with tf.variable_scope(f'{name}_psi_net', reuse=reuse):
        pi = tf.constant(np.pi)
        x_quantiles = tf.cast(tf.range(quantile_embedding_dim), tf.float32) * pi * quantiles_tiled
        x_quantiles = tf.cos(x_quantiles)
        x_quantiles = tf.layers.dense(x_quantiles, h_dim)

    return x_quantiles

def cartpole_f_net():
    with tf.variable_scope(f'{name}_f_net', reuse=reuse):
        x = self.noisy_norm_activation(x, 128, norm=None, name='noisy_relu')
        quantile_values = self.noisy(x, out_dime, name='noisy')
        quantile_values = tf.reshape(quantile_values, (n_quantiles, batch_size, out_dime))
        q = tf.reduce_mean(quantile_values, axis=0)

    return quantile_values, q