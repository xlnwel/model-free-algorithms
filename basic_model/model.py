import os, atexit, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utils import yaml_op, tf_utils
""" 
Module defines the basic functions to build a tesorflow graph
Model further defines save & restore functionns based onn Module
For example, Actor-Critic should inherit Module and DDPG should inherit Model
since we generally save parameters all together in DDPG
"""

class Module():
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph=tf.get_default_graph(),
                 reuse=None,
                 log_tensorboard=False, 
                 log_params=False,
                 device=None,
                 **kwargs):
        """ Basic module which defines the basic functions to build a tensorflow graph
        
        Arguments:
            name {str} -- Name of this module
            args {dict} -- A dictionary which specifies necessary arguments for building graph
        
        Keyword Arguments:
            graph {tf.Graph} -- The default graph which this module is built upon. Note that Module class
                                does not have authorized to change the difault graph. Graph specified here
                                is only used to acquire tensorflow variables. See @property for examples
                                (default: {tf.get_default_graph()})
            reuse {[bool or None]} -- Option for resuing variables (default: {None})
            log_tensorboard {bool} -- Option for logging information to tensorboard (default: {False})
            log_params {bool} -- Option for logging parameters to tensorboard (default: {False})
            device {[str or None]} -- Device where graph build upon {default: {None}}
        """

        self.name = name
        self._args = args
        self._graph = graph
        self._reuse = reuse
        self._log_tensorboard = log_tensorboard
        self._log_params = log_params
        self._device = device
        
        self.build_graph(device=device)
        
    def build_graph(self, **kwargs):
        if kwargs['device']:
            if 'gpu' in kwargs['device']:
                import ray
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in ray.get_gpu_ids()])
            with tf.device(kwargs['device']):
                with tf.variable_scope(self.name, reuse=self._reuse):
                    self._build_graph(**kwargs)
        else:
            with tf.variable_scope(self.name, reuse=self._reuse):
                self._build_graph(**kwargs)

    @property
    def global_variables(self):
        # _variable_scope is defined by sub-class if needed
        scope = getattr(self, '_variable_scope') if hasattr(self, '_variable_scope')  else self.name
        return self._graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    @property
    def trainable_variables(self):
        # _variable_scope is defined by sub-class if needed
        scope = getattr(self, '_variable_scope') if hasattr(self, '_variable_scope')  else self.name
        return self._graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        
    @property
    def perturbable_variables(self):
        return [var for var in self.trainable_variables if 'LayerNorm' not in var.name]
        
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
        return tc.layers.l2_regularizer(self._args['weight_decay'] if self.name in self._args and 'weight_decay' in self._args else 0.)
    
    @property
    def l2_loss(self):
        return tf.losses.get_regularization_loss(scope=self.name, name=self.name + 'l2_loss')
            
    """ Implementation """
    def _build_graph(self, **kwargs):
        raise NotImplementedError
        
    def _optimization_op(self, loss, tvars=None):
        with tf.variable_scope(self.name + '_optimizer', reuse=self._reuse):
            optimizer, global_step = self._adam_optimizer()
            grads_and_vars = self._compute_gradients(loss, optimizer, tvars=tvars)
            opt = self._apply_gradients(optimizer, grads_and_vars, global_step)

        return opt

    def _adam_optimizer(self):
        # params for optimizer
        learning_rate = float(self._args['optimizer']['learning_rate'])
        beta1 = float(self._args['optimizer']['beta1']) if 'beta1' in self._args else 0.9
        beta2 = float(self._args['optimizer']['beta2']) if 'beta2' in self._args else 0.999
        decay_rate = float(self._args['optimizer']['decay_rate']) if 'decay_rate' in self._args else 1.
        decay_steps = float(self._args['optimizer']['decay_steps']) if 'decay_steps' in self._args else 1e6
        epsilon = float(self._args['optimizer']['epsilon']) if 'epsilon' in self._args else 1e-8

        # setup optimizer
        if decay_rate == 1.:
            global_step = None
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        else:
            global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(), trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

        if self._log_tensorboard and decay_rate != 1:
            tf.summary.scalar('learning_rate_', learning_rate)

        return optimizer, global_step

    def _compute_gradients(self, loss, optimizer, tvars=None):
        clip_norm = self._args['optimizer']['clip_norm'] if 'clip_norm' in self._args else 5.
    
        update_ops = self._graph.get_collection(tf.GraphKeys.UPDATE_OPS)
        with self._graph.control_dependencies(update_ops):
            tvars = self.trainable_variables if tvars is None else tvars
            grads, tvars = list(zip(*optimizer.compute_gradients(loss, var_list=tvars)))
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        
        return list(zip(grads, tvars))

    def _apply_gradients(self, optimizer, grads_and_vars, global_step=None):
        with tf.variable_scope('learn_steps', reuse=self._reuse):
            self.learn_steps = tf.get_variable('learn_steps', shape=[], 
                                               initializer=tf.constant_initializer(), trainable=False)
            step_op = tf.assign(self.learn_steps, self.learn_steps + 1, name='update_learn_steps')

        with tf.control_dependencies([step_op]):
            opt_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        if self._log_params:
            with tf.name_scope('grads'):
                for grad, var in grads_and_vars:
                    if grad is not None:
                        tf.summary.histogram(var.name.replace(':0', ''), grad)
            with tf.name_scope('params'):
                for var in self.trainable_variables:
                    tf.summary.histogram(var.name.replace(':0', ''), var)

        return opt_op

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

    def _lstm(self, x, units, keep_prob=1.0):
        if isinstance(units, int):
            num_layers = 1
            units = [units]
        else:
            num_layers = len(units)
        
        if isinstance(keep_prob, float):
            keep_prob = [keep_prob] * num_layers
        assert len(units) == len(keep_prob), 'Dimensions of units and keep_prob do not match.'
        
        def cell(units, keep_prob):
            cell = tc.rnn.BasicLSTMCell(units)
            if keep_prob != 1.0:
                cell = tc.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

            return cell
        
        if num_layers > 1:
            cells = tc.rnn.MultiRNNCell([cell(n, keep_prob) for n in units])
    
        initial_state = cells.zero_state(x.shape[0], tf.float16)

        outputs, final_state = tf.nn.dynamic_rnn(cells, x, initial_state=initial_state)

        return outputs, initial_state, final_state

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


class Model(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args,
                 sess_config=None, 
                 reuse=None, 
                 save=True,
                 log_tensorboard=False,  
                 log_params=False,
                 log_score=False,
                 device=None,
                 **kwargs):
        """ Model, inherited from Module, further defines some boookkeeping functions,
        such as session management, save & restore operations, tensorboard loggings, and etc.
        
        Arguments:
            name {str} -- Name of the model
            args {dict} -- A dictionary which specifies necessary arguments for building graph
        
        Keyword Arguments:
            sess_config {tf.ConfigProto} -- session configuration (default: {None})
            reuse {[bool or None]} -- Option for resuing variables (default: {None})
            save {bool} -- Option for saving model (default: {True})
            log_tensorboard {bool} -- Option for logging information to tensorboard (default: {False})
            log_params {bool} -- Option for logging parameters to tensorboard (default: {False})
            log_score {bool} -- Option for logging score to tensorboard (default: {False})
            device {[str or None]} -- Device where graph build upon {default: {None}}
        """

        self._graph = tf.Graph()

        super().__init__(name, args, self._graph, reuse, log_tensorboard, 
                         log_params=log_params, device=device, **kwargs)
            
        if self._log_tensorboard:
            self.graph_summary, self.writer = self._setup_tensorboard_summary(args['tensorboard_root_dir'])
            
        # rl-specific log configuration, not in self._build_graph to avoid being included in self.graph_summary
        if self._log_tensorboard and log_score:
            self.score, self.avg_score, self.score_counter, self.score_log_op = self._setup_score_logs()

        # initialize session and global variables
        if sess_config is None:
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            # sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self._graph, config=sess_config)
        atexit.register(self.sess.close)
    
        if not self._reuse:
            self.sess.run(tf.variables_initializer(self.global_variables))
            
        if save:
            self._saver = self._setup_saver(save)
            self._model_name, self._model_dir, self._model_file = self._setup_model_path(args['model_root_dir'],
                                                                                         args['model_dir'],
                                                                                         args['model_name'])
            self.restore()
    
    @property
    def global_variables(self):
        return super().global_variables + self._graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope='scores')
        
    def build_graph(self, **kwargs):
        with self._graph.as_default():
            super().build_graph(**kwargs)

    def restore(self):
        """ To restore the most recent model, simply leave filename None
        To restore a specific version of model, set filename to the model stored in saved_models
        """
        try:
            self._saver.restore(self.sess, self._model_file)
        except:
            print('Model {}: No saved model for "{}" is found. \nStart Training from Scratch!'.format(self._model_name, self.name))
        else:
            print("Model {}: Params for {} are restored.".format(self._model_name, self.name))

    def save(self):
        if hasattr(self, '_saver'):
            self._saver.save(self.sess, self._model_file)
            yaml_op.save_args(self._args, filename=self._model_dir + '/args.yaml')

    def log_score(self, score, avg_score):
        if self._log_tensorboard:
            feed_dict = {
                self.score: score,
                self.avg_score: avg_score
            }

            score_count, summary = self.sess.run([self.score_counter, self.score_log_op], feed_dict=feed_dict)
            self.writer.add_summary(summary, score_count)

    """ Implementation """
    def _setup_score_logs(self):
        with self._graph.as_default():
            with tf.variable_scope('scores', reuse=self._reuse):
                score = tf.placeholder(tf.float32, shape=None, name='score')
                avg_score = tf.placeholder(tf.float32, shape=None, name='average_score')

                score_counter = tf.get_variable('score_counter', shape=[], initializer=tf.constant_initializer(1), trainable=False)
                step_op = tf.assign(score_counter, score_counter + 1, name='update_score_counter')
                
                score_log = tf.summary.scalar('score_', score)
                avg_score_log = tf.summary.scalar('average_score_', avg_score)

                with tf.control_dependencies([step_op]):
                    score_log_op = tf.summary.merge([score_log, avg_score_log], name='score_log_op')

        return score, avg_score, score_counter, score_log_op

    def _setup_saver(self, save):
        return tf.train.Saver(self.global_variables) if save else None

    def _setup_model_path(self, root_dir, model_dir, model_name):
        model_dir = Path(root_dir) / model_dir / model_name

        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)

        model_file = str(model_dir / model_name)

        return model_name, str(model_dir), model_file

    def _setup_tensorboard_summary(self, root_dir):
        with self._graph.as_default():
            graph_summary = tf.summary.merge_all()
            filename = os.path.join(root_dir, self._args['model_dir'], self._args['model_name'])
            writer = tf.summary.FileWriter(filename, self._graph)
            atexit.register(writer.close)

        return graph_summary, writer
