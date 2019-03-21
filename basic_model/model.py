import os, atexit, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

from utility.logger import Logger
from utility import yaml_op
from basic_model.layer import Layer
""" 
Module defines the basic functions to build a tesorflow graph
Model further defines save & restore functionns based onn Module
For example, Actor-Critic should inherit Module and DDPG should inherit Model
since we generally save parameters all together in DDPG
"""

class Module(Layer):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph=tf.get_default_graph(),
                 reuse=None,
                 log_tensorboard=False, 
                 log_params=False,
                 device=None):
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
            log_tensorboard {bool} -- Option for tensorboard setup (default: {False})
            log_params {bool} -- Option for logging network parameters to tensorboard (default: {False})
            device {[str or None]} -- Device where graph build upon {default: {None}}
        """
        self.graph = graph
        self.reuse = reuse
        self.log_tensorboard = log_tensorboard
        self.log_params = log_params
        self.device = device

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
    def global_variables(self):
        # variable_scope is defined by sub-class if needed
        scope = getattr(self, 'variable_scope') if hasattr(self, 'variable_scope')  else self.name
        return self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    @property
    def trainable_variables(self):
        # variable_scope is defined by sub-class if needed
        scope = getattr(self, 'variable_scope') if hasattr(self, 'variable_scope')  else self.name
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        
    @property
    def perturbable_variables(self):
        return [var for var in self.trainable_variables if 'LayerNorm' not in var.name]
            
    """ Implementation """
    def _build_graph(self):
        raise NotImplementedError
        
    def _optimization_op(self, loss, tvars=None, global_step=None):
        with tf.variable_scope(self.name + '_optimizer', reuse=self.reuse):
            optimizer, global_step = self._adam_optimizer(global_step=global_step)
            grads_and_vars = self._compute_gradients(loss, optimizer, tvars=tvars)
            opt = self._apply_gradients(optimizer, grads_and_vars, global_step)

        return opt, global_step

    def _adam_optimizer(self, global_step=None):
        # params for optimizer
        learning_rate = float(self.args['learning_rate'])
        beta1 = float(self.args['beta1']) if 'beta1' in self.args else 0.9
        beta2 = float(self.args['beta2']) if 'beta2' in self.args else 0.999
        decay_rate = float(self.args['decay_rate']) if 'decay_rate' in self.args else 1.
        decay_steps = float(self.args['decay_steps']) if 'decay_steps' in self.args else 1e6
        epsilon = float(self.args['epsilon']) if 'epsilon' in self.args else 1e-8

        # setup optimizer
        if global_step or decay_rate != 1.:
            global_step = tf.Variable(0, trainable=False, name='global_step')
        else:
            global_step = None

        if decay_rate == 1.:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        else:
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

            if self.log_tensorboard:
                tf.summary.scalar('learning_rate_', learning_rate)

        return optimizer, global_step

    def _compute_gradients(self, loss, optimizer, tvars=None):
        clip_norm = self.args['clip_norm'] if 'clip_norm' in self.args else 5.
    
        update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope(self.name + '_gradients'):
            with self.graph.control_dependencies(update_ops):
                tvars = tvars if tvars else self.trainable_variables
                grads, tvars = list(zip(*optimizer.compute_gradients(loss, var_list=tvars)))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        
        return list(zip(grads, tvars))

    def _apply_gradients(self, optimizer, grads_and_vars, global_step=None):
        opt_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name=self.name + '_apply_gradients')
        
        if self.log_params:
            with tf.name_scope('grads'):
                for grad, var in grads_and_vars:
                    if grad is not None:
                        tf.summary.histogram(var.name.replace(':0', ''), grad)
            with tf.name_scope('params'):
                for var in self.trainable_variables:
                    tf.summary.histogram(var.name.replace(':0', ''), var)

        return opt_op


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
                 device=None):
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

        self.graph = tf.Graph()

        super().__init__(name, args, self.graph, reuse=reuse, log_tensorboard=log_tensorboard, 
                         log_params=log_params, device=device)

        if self.log_tensorboard:
            self.logger = self._setup_logger()
            self.graph_summary, self.writer = self._setup_tensorboard_summary()
        
        # rl-specific log configuration, not in self._build_graph to avoid being included in self.graph_summary
        if log_score:
            assert self.log_tensorboard, 'Must set up tensorboard writer beforehand'
            if 'num_workers' in self.args:
                self.scores, self.avg_scores, self.score_counters, self.score_log_ops = self._setup_multiple_score_logs()
            else:
                self.score, self.avg_score, self.score_counter, self.score_log_op = self._setup_score_logs()

        # initialize session and global variables
        if sess_config is None:
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            # sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        atexit.register(self.sess.close)
    
        if not self.reuse:
            self.sess.run(tf.variables_initializer(self.global_variables))
            
        if save:
            self.saver = self._setup_saver(save)
            self.model_name, self.model_dir, self.model_file = self._setup_model_path(
                args['model_root_dir'],
                args['model_dir'],
                args['model_name']
            )
            self.restore()
    
    @property
    def global_variables(self):
        return super().global_variables + self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope='scores')
        
    def build_graph(self):
        with self.graph.as_default():
            super().build_graph()

    def restore(self):
        """ To restore the most recent model, simply leave filename None
        To restore a specific version of model, set filename to the model stored in saved_models
        """
        try:
            self.saver.restore(self.sess, self.model_file)
        except:
            print(f'Model {self.model_name}: No saved model for "{self.name}" is found. \nStart Training from Scratch!')
        else:
            print(f"Model {self.model_name}: Params for {self.name} are restored.")

    def save(self):
        if hasattr(self, 'saver'):
            self.saver.save(self.sess, self.model_file)

    def log_score(self, score, avg_score):
        feed_dict = {
            self.score: score,
            self.avg_score: avg_score
        }

        score_count, summary = self.sess.run([self.score_counter, self.score_log_op], feed_dict=feed_dict)
        self.writer.add_summary(summary, score_count)

    def log_tabular(self, key, value):
        self.logger.log_tabular(key, value)

    def dump_tabular(self, print_terminal_info=False):
        self.logger.dump_tabular(print_terminal_info=print_terminal_info)

    """ Implementation """
    def _setup_score_logs(self, name=None):
        """ score logs for a single agent """
        with self.graph.as_default():
            if name is None:
                name = 'scores'
            with tf.variable_scope(name, reuse=self.reuse):
                score = tf.placeholder(tf.float32, shape=None, name='score')
                avg_score = tf.placeholder(tf.float32, shape=None, name='average_score')

                score_counter = tf.Variable(0, trainable=False, name='score_counter')
                step_op = tf.assign(score_counter, score_counter + 1, name='update_score_counter')
                
                score_log = tf.summary.scalar('score_', score)
                avg_score_log = tf.summary.scalar('average_score_', avg_score)

                with tf.control_dependencies([step_op]):
                    score_log_op = tf.summary.merge([score_log, avg_score_log], name='score_log_op')

        return score, avg_score, score_counter, score_log_op

    def _setup_multiple_score_logs(self):
        with self.graph.as_default():
            scores = []
            avg_scores = []
            score_counters = []
            score_log_ops = []

            with tf.variable_scope('scores', reuse=self.reuse):
                for i in range(1, self.args['num_workers']+1):
                    score, avg_score, score_counter, score_log_op = self._setup_score_logs(name='worker_{}'.format(i))

                    scores.append(score)
                    avg_scores.append(avg_score)
                    score_counters.append(score_counter)
                    score_log_ops.append(score_log_op)

        return scores, avg_scores, score_counters, score_log_ops

    def _setup_saver(self, save):
        return tf.train.Saver(self.global_variables) if save else None

    def _setup_model_path(self, root_dir, model_dir, model_name):
        model_dir = Path(root_dir) / model_dir / model_name

        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)

        model_file = str(model_dir / 'ckpt')
        return model_name, str(model_dir), model_file

    def _setup_tensorboard_summary(self):
        with self.graph.as_default():
            graph_summary = tf.summary.merge_all()
            filename = os.path.join(self.args['log_root_dir'], 
                                    self.args['model_dir'], 
                                    self.args['model_name'])
            writer = tf.summary.FileWriter(filename, self.graph)
            atexit.register(writer.close)
        yaml_op.save_args(self.args, filename=filename + '/args.yaml')
        return graph_summary, writer

    def _setup_logger(self):
        log_dir = os.path.join(self.args['log_root_dir'], 
                                self.args['model_dir'], 
                                self.args['model_name'])
        
        logger = Logger(log_dir, self.name, self.args['model_name'])

        return logger
