import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.keras as tk

from utility import tf_utils


class Layer():
    def __init__(self, name, args):
        self.name = name
        self._args = args

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
        return tc.layers.l2_regularizer(self._args['weight_decay']) if self.name in self._args and 'weight_decay' in self._args else None
    
    @property
    def l2_loss(self):
        return tf.losses.get_regularization_loss(scope=self.name, name=self.name + 'l2_loss')

    """ Layers """
    def _dense(self, x, units, kernel_initializer=tf_utils.xavier_initializer(), name=None, reuse=None):
        return tf.layers.dense(x, units, kernel_initializer=kernel_initializer, 
                               kernel_regularizer=self.l2_regularizer, 
                               trainable=self.trainable, 
                               name=name, reuse=reuse)

    def _dense_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                               normalization=tc.layers.layer_norm, activation=tf.nn.relu, name=None, reuse=None):
        def layer_imp():
            y = self._dense(x, units, kernel_initializer=kernel_initializer, reuse=reuse)
            y = tf_utils.norm_activation(y, normalization=normalization, activation=activation, 
                                        training=self.training, trainable=self.trainable)

            return y

        x = self._wrap_layer(name, reuse, layer_imp)

        return x

    def _dense_resnet(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(), 
                      normalization=tc.layers.layer_norm, name=None, reuse=None):
        """
        kernel_initializer specifies the initialization of the last layer in the residual module
        relu is used as the default activation and no designation is allowed
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        name = self._get_name(name, 'dense_resnet')

        with tf.variable_scope(name, reuse=reuse):
            y = tf_utils.norm_activation(x, normalization=normalization, activation=tf.nn.relu, training=self.training)
            y = self._dense_norm_activation(y, units, kernel_initializer=tf_utils.kaiming_initializer(), 
                                            normalization=normalization, activation=tf.nn.relu, reuse=reuse)
            y = self._dense(y, units, kernel_initializer=kernel_initializer, reuse=reuse)
            x += y

        return x

    def _dense_resnet_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer() ,
                                      normalization=tc.layers.layer_norm, 
                                      activation=tf.nn.relu, name=None, reuse=None):
        """
        normalization is used in both the last layer in the residual module and 
        the layer immediately following the residual module
        activation is used only in the layer immediately following the residual module
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        def layer_imp():
            y = self._dense_resnet(x, units, kernel_initializer, normalization, reuse=reuse)
            y = tf_utils.norm_activation(y, normalization, activation)

            return y
        
        x = self._wrap_layer(name, reuse, layer_imp)

        return x

    def _conv(self, x, filters, kernel_size, strides=1, padding='same', 
              kernel_initializer=tf_utils.xavier_initializer(), name=None, reuse=None): 
        return tf.layers.conv2d(x, filters, kernel_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer, 
                                kernel_regularizer=self.l2_regularizer, 
                                trainable=self.trainable, name=name, reuse=reuse)

    def _conv_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                              kernel_initializer=tf_utils.kaiming_initializer(), 
                              normalization=tf.layers.batch_normalization, 
                              activation=tf.nn.relu, name=None, reuse=None):
        def layer_imp():
            y = self._conv(x, filters, kernel_size, 
                            strides=strides, padding=padding, 
                            kernel_initializer=kernel_initializer, 
                            reuse=reuse)
            y = tf_utils.norm_activation(y, normalization=normalization, activation=activation, 
                                            training=self.training, trainable=self.trainable)
            
            return y

        x = self._wrap_layer(name, reuse, layer_imp)

        return x
    
    def _conv_resnet(self, x, filters, kernel_size, strides=1, padding='same', 
                     kernel_initializer=tf_utils.kaiming_initializer(),
                     normalization=tf.layers.batch_normalization, name=None, reuse=None):
        """
        kernel_initializer specifies the initialization of the last layer in the residual module
        relu is used as the default activation and no designation is allowed
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        name = self._get_name(name, 'conv_resnet')

        with tf.variable_scope(name, reuse=reuse):
            y = tf_utils.norm_activation(x, normalization=normalization, activation=tf.nn.relu, training=self.training)
            y = self._conv_norm_activation(y, filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                           kernel_initializer=tf_utils.kaiming_initializer(), 
                                           normalization=normalization, activation=tf.nn.relu, reuse=reuse)
            y = self._conv(y, filters, kernel_size, strides=strides, padding=padding,
                           kernel_initializer=kernel_initializer, reuse=reuse)
            x += y

        return x
    
    def _conv_resnet_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                                     kernel_initializer=tf_utils.kaiming_initializer(),
                                     normalization=tf.layers.batch_normalization, activation=tf.nn.relu, name=None, reuse=None):
        """
        normalization is used in both the last layer in the residual module and 
        the layer immediately following the residual module
        activation is used only in the layer immediately following the residual module
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        def layer_imp():
            y = self._conv_resnet(x, filters, kernel_size, 
                                  strides=strides, padding=padding, 
                                  kernel_initializer=kernel_initializer,
                                  normalization=normalization, reuse=reuse)
            y = tf_utils.norm_activation(y, normalization=normalization, activation=activation, 
                                            training=self.training, trainable=self.trainable)

            return y
        
        x = self._wrap_layer(name, reuse, layer_imp)

        return x

    def _convtrans(self, x, filters, kernel_size, strides=1, padding='same', 
                   kernel_initializer=tf_utils.xavier_initializer(), name=None, reuse=None): 
        return tf.layers.conv2d_transpose(x, filters, kernel_size, 
                                          strides=strides, padding=padding, 
                                          kernel_initializer=kernel_initializer, 
                                          kernel_regularizer=self.l2_regularizer, 
                                          trainable=self.trainable, name=name, reuse=reuse)

    def _convtrans_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                                   kernel_initializer=tf_utils.kaiming_initializer(), 
                                   normalization=tf.layers.batch_normalization, 
                                   activation=tf.nn.relu, name=None, reuse=None):
        def layer_imp():
            y = self._convtrans(x, filters, kernel_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer,
                                reuse=reuse)
            y = tf_utils.norm_activation(y, normalization=normalization, activation=activation, 
                                            training=self.training, trainable=self.trainable)

            return y

        x = self._wrap_layer(name, reuse, layer_imp)

        return x

    def _noisy(self, x, units, kernel_initializer=tf_utils.xavier_initializer(), 
               name=None, reuse=None, sigma=.4):
        name = self._get_name(name, 'noisy')
        
        with tf.variable_scope(name, reuse=reuse):
            y = self._dense(x, units, kernel_initializer=kernel_initializer, reuse=reuse)
            
            with tf.variable_scope('noisy', reuse=reuse):
                # params for the noisy layer
                features = x.shape.as_list()[-1]
                w_shape = [features, units]
                b_shape = [units]
                epsilon_w = tf.truncated_normal(w_shape, stddev=sigma, name='epsilon_w')
                epsilon_b = tf.truncated_normal(b_shape, stddev=sigma, name='epsilon_b')
                noisy_w = tf.get_variable('noisy_w', shape=w_shape, 
                                          initializer=kernel_initializer,
                                          regularizer=self.l2_regularizer, 
                                          trainable=self.trainable)
                noisy_b = tf.get_variable('noisy_b', shape=b_shape, 
                                          initializer=tf.constant_initializer(sigma / np.sqrt(units)), 
                                          trainable=self.trainable)
                
                # output of the noisy layer
                x = tf.matmul(x, noisy_w * epsilon_w) + noisy_b * epsilon_b

            x = x + y

        if self.trainable:
            return x
        else:
            return y

    def _noisy_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                               normalization=tc.layers.layer_norm, activation=tf.nn.relu, 
                               name=None, reuse=None, sigma=.4):
        def layer_imp():
            y = self._noisy(x, units, kernel_initializer=kernel_initializer, 
                            name=name, reuse=reuse, sigma=sigma)
            y = tf_utils.norm_activation(y, normalization=normalization, activation=activation, 
                                         training=self.training, trainable=self.trainable)
            
            return y

        x = self._wrap_layer(name, reuse, layer_imp)

        return x

    def _noisy_resnet(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                      normalization=tc.layers.layer_norm, name=None, reuse=None, sigma=.4):
        """
        kernel_initializer specifies the initialization of the last layer in the residual module
        relu is used as the default activation and no designation is allowed
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        name = self._get_name(name, 'noisy_resnet')

        with tf.variable_scope(name, reuse=reuse):
            y = tf_utils.norm_activation(x, normalization=normalization, activation=tf.nn.relu, 
                                         training=self.training, trainable=self.trainable)
            y = self._noisy_norm_activation(y, units, kernel_initializer=tf_utils.kaiming_initializer(), 
                                            normalization=normalization, activation=tf.nn.relu, sigma=sigma)
            y = self._noisy(y, units, kernel_initializer=kernel_initializer, reuse=reuse, sigma=sigma)
            x += y

        return x
    
    def _noisy_resnet_norm_activation(self, x, units, kernel_initializer=tf_utils.kaiming_initializer(),
                                      normalization=tc.layers.layer_norm, activation=tf.nn.relu, 
                                      name=None, reuse=None, sigma=.4):
        """
        normalization is used in both the last layer in the residual module and 
        the layer immediately following the residual module
        activation is used only in the layer immediately following the residual module
        
        Caution: _reset_counter should be called first if this residual module is reused
        """
        def layer_imp():
            y = self._noisy_resnet(x, units, kernel_initializer, normalization, reuse=reuse, sigma=sigma)
            y = tf_utils.norm_activation(y, normalization=normalization, activation=activation, 
                                         training=self.training, trainable=self.trainable)

            return y
        
        x = self._wrap_layer(name, reuse, layer_imp)

        return x

    def _lstm(self, x, units, initial_state=None, return_cell=False):
        if isinstance(units, int):
            num_layers = 1
            units = [units]
        else:
            num_layers = len(units)
        
        if num_layers == 1:
            lstm = tk.layers.CuDNNLSTM(units[0])
        else:
            cells = [tk.layers.CuDNNLSTM(n) for n in units]
            lstm = tk.layers.StackedRNNCells(cells)

        x = lstm(x, initial_state=initial_state)

        if return_cell:
            return x, lstm
        else:
            return x

    """ Auxiliary functions """
    def _reset_counter(self, name):
        counter = name + '_counter'
        # assert hasattr(self, counter), 'No counter named {}'.format(counter)
        setattr(self, counter, -1)   # to avoid scope name conflict caused by _dense_resnet_norm_activation

    def _get_name(self, name, default_name):
        if name is None:
            name_counter = default_name + '_counter'
            if hasattr(self, name_counter):
                setattr(self, name_counter, getattr(self, name_counter) + 1)
            else:
                setattr(self, name_counter, 0)
            name = '{}_{}'.format(default_name, getattr(self, name_counter))

        return name

    def _wrap_layer(self, name, reuse, layer_imp):
        if name:
            with tf.variable_scope(name, reuse=reuse):
                x = layer_imp()
        else:
            x = layer_imp()

        return x
