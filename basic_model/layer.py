import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.keras as tk

from utility import tf_utils
from utility.utils import pwc
from utility.debug_tools import assert_colorize


class Layer():
    def __init__(self, name, args):
        self.name = name
        self.args = args

    @property
    def training(self):
        """ this property should only be used with batch normalization, 
        self._training should be a boolean placeholder """
        return getattr(self, '_training', False)

    @property
    def trainable(self):
        return getattr(self, '_trainable', True)

    @property
    def l2_regularizer(self):
        return (tk.regularizers.l2(self.args['weight_decay']) 
                if 'weight_decay' in self.args and self.args['weight_decay'] > 0
                else None)
    
    @property
    def l2_loss(self):
        return tf.losses.get_regularization_loss(scope=self.name, name=self.name + 'l2_loss')

    """ Layers
    The main reason why we define layers as member functions is 
    that we want to  automatically handle l2 regularization.
    """
    def dense(self, x, units, kernel_initializer=tf_utils.xavier_initializer(), name=None):
        return tf.layers.dense(x, units, kernel_initializer=kernel_initializer, 
                               kernel_regularizer=self.l2_regularizer, 
                               name=name)

    def dense_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                               norm=tc.layers.layer_norm, activation=tf.nn.relu, name=None):
        def layer_imp():
            y = self.dense(x, units, kernel_initializer=kernel_initializer)
            y = tf_utils.norm_activation(y, norm=norm, activation=activation, 
                                        training=self.training)

            return y

        x = tf_utils.wrap_layer(name, layer_imp)

        return x

    def dense_resnet(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(), 
                      norm=tc.layers.layer_norm, name=None):
        """
        kernel_initializer specifies the initialization of the last layer in the residual module
        relu is used as the default activation and no designation is allowed
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        name = self.get_name(name, 'dense_resnet')

        with tf.variable_scope(name):
            y = tf_utils.norm_activation(x, norm=norm, activation=tf.nn.relu, training=self.training)
            y = self.dense_norm_activation(y, units, kernel_initializer=kernel_initializer, 
                                            norm=norm, activation=tf.nn.relu)
            y = self.dense(y, units, kernel_initializer=kernel_initializer)
            x += y

        return x

    def dense_resnet_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer() ,
                                      norm=tc.layers.layer_norm, 
                                      activation=tf.nn.relu, name=None):
        """
        normalization is used in both the last layer in the residual module and 
        the layer immediately following the residual module
        activation is used only in the layer immediately following the residual module
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        def layer_imp():
            y = self.dense_resnet(x, units, kernel_initializer, norm)
            y = tf_utils.norm_activation(y, norm, activation)

            return y
        
        x = tf_utils.wrap_layer(name, layer_imp)

        return x

    def conv(self, x, filters, kernel_size, strides=1, padding='same', 
              kernel_initializer=tf_utils.xavier_initializer(), name=None): 
        if padding != 'same' and padding != 'valid':
            x = tf_utils.padding(x, kernel_size // 2, kernel_size // 2, mode=padding)
            padding = 'valid'

        return tf.layers.conv2d(x, filters, kernel_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer, 
                                kernel_regularizer=self.l2_regularizer, 
                                name=name)

    def conv_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                              kernel_initializer=tf_utils.kaiming_initializer(), 
                              norm=tf.layers.batch_normalization, 
                              activation=tf.nn.relu, name=None):
        def layer_imp():
            y = self.conv(x, filters, kernel_size, 
                            strides=strides, padding=padding, 
                            kernel_initializer=kernel_initializer)
            y = tf_utils.norm_activation(y, norm=norm, activation=activation, 
                                            training=self.training)
            
            return y

        x = tf_utils.wrap_layer(name, layer_imp)

        return x
    
    def conv_resnet(self, x, filters, kernel_size, strides=1, padding='same', 
                     kernel_initializer=tf_utils.kaiming_initializer(),
                     norm=tf.layers.batch_normalization, name=None):
        """
        kernel_initializer specifies the initialization of the last layer in the residual module
        relu is used as the default activation and no designation is allowed
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        name = self.get_name(name, 'conv_resnet')

        with tf.variable_scope(name):
            y = tf_utils.norm_activation(x, norm=norm, activation=tf.nn.relu, training=self.training, name='NormAct')
            y = self.conv_norm_activation(y, filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                           kernel_initializer=kernel_initializer, 
                                           norm=norm, activation=tf.nn.relu)
            y = self.conv(y, filters, kernel_size, strides=strides, padding=padding,
                           kernel_initializer=kernel_initializer)
            x += y

        return x
    
    def conv_resnet_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                                     kernel_initializer=tf_utils.kaiming_initializer(),
                                     norm=tf.layers.batch_normalization, activation=tf.nn.relu, name=None):
        """
        normalization is used in both the last layer in the residual module and 
        the layer immediately following the residual module
        activation is used only in the layer immediately following the residual module
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        def layer_imp():
            y = self.conv_resnet(x, filters, kernel_size, 
                                  strides=strides, padding=padding, 
                                  kernel_initializer=kernel_initializer,
                                  norm=norm)
            y = tf_utils.norm_activation(y, norm=norm, activation=activation, 
                                            training=self.training)

            return y
        
        x = tf_utils.wrap_layer(name, layer_imp)

        return x

    def convtrans(self, x, filters, kernel_size, strides=1, padding='same', 
                   kernel_initializer=tf_utils.xavier_initializer(), name=None): 
        return tf.layers.conv2d_transpose(x, filters, kernel_size, 
                                          strides=strides, padding=padding, 
                                          kernel_initializer=kernel_initializer, 
                                          kernel_regularizer=self.l2_regularizer, 
                                          name=name)

    def convtrans_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                                   kernel_initializer=tf_utils.kaiming_initializer(), 
                                   norm=tf.layers.batch_normalization, 
                                   activation=tf.nn.relu, name=None):
        def layer_imp():
            y = self.convtrans(x, filters, kernel_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer)
            y = tf_utils.norm_activation(y, norm=norm, activation=activation, 
                                            training=self.training)

            return y

        x = tf_utils.wrap_layer(name, layer_imp)

        return x

    def noisy(self, x, units, kernel_initializer=tf_utils.xavier_initializer(distribution='uniform'), 
               name=None, sigma=.4):
        """ noisy layer using factorized Gaussian noise """
        name = self.get_name(name, 'noisy')
        
        with tf.variable_scope(name):
            y = self.dense(x, units, kernel_initializer=kernel_initializer)
            
            with tf.variable_scope('noisy'):
                # params for the noisy layer
                features = x.shape.as_list()[-1]
                w_in_dim = [features, 1]
                w_out_dim = [1, units]
                w_shape = [features, units]
                b_shape = [units]

                epsilon_w_in = tf.random.truncated_normal(w_in_dim, stddev=sigma)
                epsilon_w_in = tf.math.sign(epsilon_w_in) * tf.math.sqrt(tf.math.abs(epsilon_w_in))
                epsilon_w_out = tf.random.truncated_normal(w_out_dim, stddev=sigma)
                epsilon_w_out = tf.math.sign(epsilon_w_out) * tf.math.sqrt(tf.math.abs(epsilon_w_out))
                epsilon_w = tf.matmul(epsilon_w_in, epsilon_w_out, name='epsilon_w')
                epsilon_b = tf.reshape(epsilon_w_out, b_shape)
                
                noisy_w = tf.get_variable('noisy_w', shape=w_shape, 
                                          initializer=kernel_initializer,
                                          regularizer=self.l2_regularizer)
                noisy_b = tf.get_variable('noisy_b', shape=b_shape, 
                                          initializer=tf.constant_initializer(sigma / np.sqrt(units)))
                
                # output of the noisy layer
                x = tf.matmul(x, noisy_w * epsilon_w) + noisy_b * epsilon_b

            x = x + y

        return x

    def noisy2(self, x, units, kernel_initializer=tf_utils.xavier_initializer(), 
               name=None, sigma=.4):
        name = self.get_name(name, 'noisy')
        
        with tf.variable_scope(name):
            y = self.dense(x, units, kernel_initializer=kernel_initializer)
            
            with tf.variable_scope('noisy'):
                # params for the noisy layer
                features = x.shape.as_list()[-1]
                w_shape = [features, units]
                b_shape = [units]

                epsilon_w = tf.random.truncated_normal(w_shape, stddev=sigma, name='epsilon_w')
                epsilon_b = tf.random.truncated_normal(b_shape, stddev=sigma, name='epsilon_b')

                noisy_w = tf.get_variable('noisy_w', shape=w_shape, 
                                          initializer=kernel_initializer,
                                          regularizer=self.l2_regularizer)
                noisy_b = tf.get_variable('noisy_b', shape=b_shape, 
                                          initializer=tf.constant_initializer(sigma / np.sqrt(units)))
                
                # output of the noisy layer
                x = tf.matmul(x, noisy_w * epsilon_w) + noisy_b * epsilon_b

            x = x + y

        return x

    def noisy_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                               norm=tc.layers.layer_norm, activation=tf.nn.relu, 
                               name=None, sigma=.4):
        def layer_imp():
            y = self.noisy(x, units, kernel_initializer=kernel_initializer, 
                            name=name, sigma=sigma)
            y = tf_utils.norm_activation(y, norm=norm, activation=activation, 
                                         training=self.training)
            
            return y

        x = tf_utils.wrap_layer(name, layer_imp)

        return x

    def noisy_resnet(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                      norm=tc.layers.layer_norm, name=None, sigma=.4):
        """
        kernel_initializer specifies the initialization of the last layer in the residual module
        relu is used as the default activation and no designation is allowed
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        name = self.get_name(name, 'noisy_resnet')

        with tf.variable_scope(name):
            y = tf_utils.norm_activation(x, norm=norm, activation=tf.nn.relu, 
                                         training=self.training)
            y = self.noisy_norm_activation(y, units, kernel_initializer=kernel_initializer, 
                                            norm=norm, activation=tf.nn.relu, sigma=sigma)
            y = self.noisy(y, units, kernel_initializer=kernel_initializer, sigma=sigma)
            x += y

        return x
    
    def noisy_resnet_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                                      norm=tc.layers.layer_norm, activation=tf.nn.relu, 
                                      name=None, sigma=.4):
        """
        normalization is used in both the last layer in the residual module and 
        the layer immediately following the residual module
        activation is used only in the layer immediately following the residual module
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        def layer_imp():
            y = self.noisy_resnet(x, units, kernel_initializer, norm, sigma=sigma)
            y = tf_utils.norm_activation(y, norm=norm, activation=activation, 
                                         training=self.training)

            return y
        
        x = tf_utils.wrap_layer(name, layer_imp)

        return x

    def lstm(self, x, units, return_sequences=False):
        assert_colorize(len(x.shape.as_list()) == 3, f'Imput Shape Error: desire shape of dimension 3, get {len(x.shape.as_list())}')
        lstm_cell = tk.layers.CuDNNLSTM(units, return_sequences=return_sequences, return_state=True)
        
        initial_state = lstm_cell.get_initial_state(x)
        x, h, c = lstm_cell(x, initial_state=initial_state)
        final_state = (h, c)

        return x, (initial_state, final_state)

    def lstm_norm(self, x, units, masks, norm=True):
        kernel_initializer = tf_utils.kaiming_initializer() if norm else tf_utils.xavier_initializer()
        xw_shape = [x.shape.as_list()[-1], units]
        xb_shape = [units]
        hw_shape = [units, units]
        hb_shape = [units]
        
        n_batch, n_steps = x.shape.as_list()[:2]

        ln = tc.layers.layer_norm

        with tf.variable_scope('lstm_norm'):
            x_w = tf.get_variable('x_w', shape=xw_shape, 
                                  initializer=kernel_initializer,
                                  regularizer=self.l2_regularizer)
            x_b = tf.get_variable('x_b', shape=xb_shape, 
                                  initializer=tf.constant_initializer(0))
            
            h_w = tf.get_variable('h_w', shape=hw_shape, 
                                  initializer=kernel_initializer,
                                  regularizer=self.l2_regularizer)
            h_b = tf.get_variable('h_b', shape=hb_shape, 
                                  initializer=tf.constant_initializer(0))

            initial_state = tf.zeros([n_batch, 2*units], name='initial_state')
            c, c = tf.split(value=initial_state, num_or_size_splits=2, axis=1)
            xs = [tf.squeeze(v, [1]) for v in tf.split(value=x, num_or_size_splits=n_steps, axis=1)]
            for idx, (x, m) in enumerate(zip(xs, masks)):
                c *= 1-masks
                h *= 1-masks
                z = ln(tf.matmul(x, x_w) + x_b) + ln(tf.matmul(h, h_w) + h_b)
                f, i, o, u = tf.split(value=z, num_or_size_splits=4, axis=1)
                f = tf.nn.sigmoid(f)
                i = tf.nn.sigmoid(i)
                o = tf.nn.sigmoid(o)
                u = tf.tanh(u)
                c = f * c + i * u
                h = o * tf.tanh(ln(c))
                xs[idx] = h
            
            final_state = (h, c)
            xs = tf.stack(xs, 1)

        return xs, (initial_state, final_state)

    """ Auxiliary functions """
    def reset_counter(self, name):
        counter = name + '_counter'
        setattr(self, counter, -1)   # to avoid scope name conflict caused by _dense_resnet_norm_activation

    def get_name(self, name, default_name):
        if name is None:
            name_counter = default_name + '_counter'
            if hasattr(self, name_counter):
                setattr(self, name_counter, getattr(self, name_counter) + 1)
            else:
                setattr(self, name_counter, 0)
            name = '{}_{}'.format(default_name, getattr(self, name_counter))

        return name

