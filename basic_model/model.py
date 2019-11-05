import os, atexit, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import tensorflow as tf

from utility.utils import isscalar
from utility.tf_utils import wrap_scope
from utility.display import assert_colorize, pwc, display_var_info
from utility.logger import Logger
from basic_model.layer import Layer

""" 
Module defines the basic functions to build a tesorflow graph
Model further defines save & restore functionns based on Module
For example, Actor-Critic should inherit Module and DDPG should inherit Model
since we generally save parameters all together in DDPG
"""

class Module(Layer):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph=tf.compat.v1.get_default_graph(),
                 scope_prefix='',
                 log_tensorboard=False, 
                 log_params=False,
                 device=None,
                 reuse=None):
        """ Basic module which defines the basic functions to build a tensorflow graph
        
        Arguments:
            name {str} -- Name of this module
            args {dict} -- A dictionary which specifies necessary arguments for building graph
        
        Keyword Arguments:
            graph {tf.Graph} -- The default graph which this module is built upon. Note that Module class
                                does not have authorized to change the difault graph. Graph specified here
                                is only used to acquire tensorflow variables. See @property for examples
                                (default: {tf.compat.v1.get_default_graph()})
            log_tensorboard {bool} -- Option for tensorboard setup (default: {False})
            log_params {bool} -- Option for logging network parameters to tensorboard (default: {False})
            device {[str or None]} -- Device where graph build upon {default: {None}}
        """
        self.graph = graph
        self.log_tensorboard = log_tensorboard
        self.log_params = log_params
        self.device = device
        self.reuse = reuse

        self.variable_scope = self._get_variable_scope(scope_prefix, name)

        super().__init__(name, args)

        self.build_graph()
        
    def build_graph(self):
        if self.device:
            with tf.device(self.device):
                with tf.variable_scope(self.name, reuse=self.reuse):
                    self._build_graph()
        else:
            with tf.variable_scope(self.name, reuse=self.reuse):
                self._build_graph()

    @property
    def scope(self):
        return getattr(self, 'variable_scope') if hasattr(self, 'variable_scope')  else self.name

    @property
    def global_variables(self):
        # variable_scope is defined by sub-class if needed
        return self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def trainable_variables(self):
        # variable_scope is defined by sub-class if needed
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        
    @property
    def perturbable_variables(self):
        return [var for var in self.trainable_variables if 'LayerNorm' not in var.name]
            
    """ Implementation """
    def _build_graph(self):
        raise NotImplementedError
        
    def _optimization_op(self, loss, tvars=None, opt_step=None, schedule_lr=False, name=None):
        with tf.device('/CPU:0'):
            with tf.variable_scope((name or self.name) + '_optimizer'):
                optimizer, learning_rate, opt_step = self._adam_optimizer(opt_step=opt_step, schedule_lr=schedule_lr, name=name)
                grads_and_vars = self._compute_gradients(loss, optimizer, tvars=tvars)
                opt_op = self._apply_gradients(optimizer, grads_and_vars, opt_step)

        return optimizer, learning_rate, opt_step, grads_and_vars, opt_op

    def _adam_optimizer(self, opt_step=None, schedule_lr=False, name=None):
        # params for optimizer
        if not schedule_lr:
            learning_rate = float(self.args['learning_rate']) if name is None else float(self.args[f'{name}_lr'])
            decay_rate = float(self.args['decay_rate']) if 'decay_rate' in self.args else 1.
            decay_steps = float(self.args['decay_steps']) if 'decay_steps' in self.args else 1e6
        beta1 = float(self.args['beta1']) if 'beta1' in self.args else 0.9
        beta2 = float(self.args['beta2']) if 'beta2' in self.args else 0.999
        epsilon = float(self.args['epsilon']) if 'epsilon' in self.args else 1e-8

        # setup optimizer
        if opt_step or 'decay_steps' in self.args:
            opt_step = tf.Variable(0, trainable=False, name='opt_step')
        else:
            opt_step = None

        if schedule_lr:
            learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')
        elif decay_rate != 1.:
            learning_rate = tf.train.exponential_decay(learning_rate, opt_step, decay_steps, decay_rate, staircase=True)
        if self.log_tensorboard and not isinstance(learning_rate, float):
            tf.compat.v1.summary.scalar('learning_rate_', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

        return optimizer, learning_rate, opt_step

    def _compute_gradients(self, loss, optimizer, tvars=None):
        clip_norm = self.args['clip_norm'] if 'clip_norm' in self.args else 5.
    
        update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope(self.name + '_gradients'):
            with self.graph.control_dependencies(update_ops):
                tvars = tvars if tvars else self.trainable_variables
                grads_vars = optimizer.compute_gradients(loss, var_list=tvars)
                # clip by global norm
                grads, tvars = zip(*grads_vars)
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                
                return list(zip(grads, tvars))
                # clip by norm
                # for i, (grad, var) in enumerate(grads_vars):
                #     if grad is not None:
                #         grads_vars[i] = (tf.clip_by_norm(grad, clip_norm), var)
        
                # return grads_vars

    def _apply_gradients(self, optimizer, grads_and_vars, opt_step=None):
        opt_op = optimizer.apply_gradients(grads_and_vars, global_step=opt_step, name=self.name + '_apply_gradients')
        
        if self.log_params:
            with tf.name_scope('grads'):
                for grad, var in grads_and_vars:
                    if grad is not None:
                        tf.summary.histogram(var.name.replace(':0', ''), grad)
            with tf.name_scope('params'):
                for var in self.trainable_variables:
                    tf.summary.histogram(var.name.replace(':0', ''), var)

        return opt_op

    def _get_variable_scope(self, scope_prefix, name):
        return f'{scope_prefix}/{name}' if scope_prefix else name


class Model(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args,
                 sess_config=None, 
                 save=False, 
                 log=False,
                 log_tensorboard=False,
                 log_params=False,
                 log_stats=False,
                 device=None,
                 reuse=None,
                 graph=None):
        """ Model, inherited from Module, further defines some boookkeeping functions,
        such as session management, save & restore operations, tensorboard loggings, and etc.
        
        Arguments:
            name {str} -- Name of the model
            args {dict} -- A dictionary which specifies necessary arguments for building graph
        
        Keyword Arguments:
            sess_config {tf.ConfigProto} -- session configuration (default: {None})
            save {bool} -- Option for saving model (default: {True})
            log {bool} -- Option for logging info using logger (default: {False})
            log_tensorboard {bool} -- Option for logging information to tensorboard (default: {False})
            log_params {bool} -- Option for logging parameters to tensorboard (default: {False})
            log_stats {bool} -- Option for logging score to tensorboard (default: {False})
            device {[str or None]} -- Device where graph build upon {default: {None}}
            reuse {bool} -- Option for reusing graph {default: {None}}
            graph {tf.Graph} -- tensorflow graph. Reconstruct a new one if None {default: {None}}
        """

        self.graph = graph if graph else tf.Graph()

        # initialize session and global variables
        if sess_config is None:
            sess_config = tf.ConfigProto(#intra_op_parallelism_threads=1,
                                        #inter_op_parallelism_threads=1,
                                        allow_soft_placement=True)
            # sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        atexit.register(self.sess.close)

        super().__init__(name, args, self.graph, log_tensorboard=log_tensorboard, 
                         log_params=log_params, device=device, reuse=reuse)

        display_var_info(self.trainable_variables)

        self.model_name = args['model_name']

        # set up logging
        log_dir = os.path.join(args['log_root_dir'], self.model_name)
        
        if log:
            self.logger = self._setup_logger(log_dir, self.model_name)
            self.logger.save_args(self.args)

        if self.log_tensorboard:
            self.graph_summary= self._setup_tensorboard_summary()
        
        if log_tensorboard or log_stats:
            self.writer = self._setup_writer(log_dir)

        self.sess.run(tf.variables_initializer(self.global_variables))

        if save:
            self.saver = self._setup_saver()
            self.model_file = self._setup_model_path(args['model_root_dir'], self.model_name)

        self.print_construction_complete()
        
    @property
    def global_variables(self):
        return super().global_variables + self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope='stats')
        
    def build_graph(self):
        with self.graph.as_default():
            super().build_graph()

    def restore(self, model_file=None):
        """
        To restore a specific version of model, set filename to the model stored in saved_models
        """
        if model_file:
            self.model_file = model_file
        if not hasattr(self, 'saver'):
            self.saver = self._setup_saver()
        try:
            ckpt = tf.train.latest_checkpoint(self.model_file)
            self.saver.restore(self.sess, ckpt)
        except:
            pwc(f'Model {self.model_name}: no saved model for "{self.name}" is found at "{self.model_file}"!',
                'magenta')
            import sys
            sys.exit()
        else:
            pwc(f'Model {self.model_name}: Params for {self.name} are restored from "{self.model_file}".', 'magenta')

    def save(self, step=None, message=''):
        if hasattr(self, 'saver'):
            path = self.saver.save(self.sess, self.model_file, global_step=step)
            if message:
                message = f'\n{message}'
            pwc(f'Model saved at {path}{message}', 'magenta')
        else:
            # name intention to treat name saver as an error, just print a warning message
            pwc('name saver is available', 'magenta')

    def record_stats(self, kwargs):
        assert isinstance(kwargs, dict)
        self._record_stats_impl(kwargs.copy())

    def store(self, **kwargs):
        self.logger.store(**kwargs)

    def get_stored_stats(self):
        return self.logger.get_stats()

    def log_tabular(self, key, value=None, mean=True, std=False, min=False, max=False):
        self.logger.log_tabular(key, value, mean, std, min, max)

    def dump_tabular(self, print_terminal_info=True):
        self.logger.dump_tabular(print_terminal_info=print_terminal_info)

    def print_construction_complete(self):
        pwc(f'{self.name} has been constructed', 'cyan')
        
    """ Implementation """
    def _setup_saver(self):
        return tf.compat.v1.train.Saver(self.global_variables)

    def _setup_logger(self, log_dir, model_name):
        return Logger(log_dir, exp_name=model_name)

    def _setup_writer(self, writer_dir):
        writer = tf.summary.FileWriter(writer_dir, self.graph)
        atexit.register(writer.close)
        
        return writer

    def _setup_model_path(self, root_dir, model_name):
        model_dir = Path(root_dir) / model_name

        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)

        model_file = str(model_dir / 'ckpt')
        return model_file

    def _setup_tensorboard_summary(self):
        with self.graph.as_default():
            graph_summary = tf.summary.merge_all()

        return graph_summary

    def _setup_stats_logs(self, stats_info, name=None):
        """
        stats_info are not in self._build_graph to avoid being included in self.graph_summary
        """
        def setup():
            stats = {}
            stats['counter'] = counter = tf.Variable(0, trainable=False, name='counter')
            step_op = tf.assign(counter, counter + 1, name='counter_update')
            merge_inputs = []
            for info in stats_info:
                stats[info] = info_ph = tf.placeholder(tf.float32, name=info)
                stats[f'{info}_log'] = log = tf.compat.v1.summary.scalar(f'{info}_', info_ph)
                merge_inputs.append(log)
            
            with tf.control_dependencies([step_op]):
                stats['log_op'] = tf.summary.merge(merge_inputs, name='log_op')

            self.sess.run(tf.variables_initializer([counter]))

            return stats

        """ Function Body """
        with self.graph.as_default():
            with tf.variable_scope('stats'):
                # stats logs for each worker
                return wrap_scope(name, lambda: setup())

    def _record_stats_impl(self, kwargs):
        # if step appears in kwargs, use it when adding summary to tensorboard
        if 'steps' in kwargs:
            step = kwargs['steps']
            del kwargs['steps']
        elif 'Steps' in kwargs:
            step = kwargs['Steps']
            del kwargs['Steps']
        else:
            step = None

        name = kwargs['name'] if 'name' in kwargs else None
        kwargs = dict([(k, v) for k, v in kwargs.items() if isscalar(v)])

        if not hasattr(self, 'stats'):
            if name is None:
                self.stats = stats = self._setup_stats_logs(kwargs.keys())
            else:
                self.stats = defaultdict(dict)
                self.stats[name] = stats = self._setup_stats_logs(kwargs.keys(), name)
        else:
            stats = self.stats[name] if name else self.stats

        feed_dict = {}

        [feed_dict.update({stats[k]: v}) for k, v in kwargs.items() if isscalar(v)]

        score_count, summary = self.sess.run([stats['counter'], stats['log_op']], 
                                            feed_dict=feed_dict)

        self.writer.add_summary(summary, step or score_count)
