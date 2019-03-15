import tensorflow as tf

from basic_model.off_policy import OffPolicy
from rainbow.networks import QNets
from utility.losses import huber_loss


class Agent(OffPolicy):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 reuse=None, 
                 save=True, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_score=False, 
                 device=None):
        # optional improvements
        options = args['Q']
        self.n_steps = options['n_steps']
        self.distributional = options['distributional']
        self.duel = options['duel']
        self.critic_loss_type = options['loss_type']

        super().__init__(name,
                         args,
                         env_args,
                         buffer_args,
                         sess_config=sess_config,
                         reuse=reuse,
                         save=save,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_score=log_score,
                         device=device)

    def _build_graph(self):
        if 'gpu' in self.device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)

        self.Q_nets = self._create_nets(self.data)
        self.action = self.Q_nets.action

        self.priority, self.Q_loss = self._loss(self.data, self.Q_nets)

        self.opt_op = self.Q_nets._optimization_op(self.Q_loss)

        self._log_loss()

    def _create_nets(self, data):
        scope_prefix = self.name

        Q_nets = QNets('QNets', 
                        self.args['Q'], 
                        self.graph, 
                        self.data, 
                        self.action_dim,
                        reuse=self.reuse,
                        scope_prefix=scope_prefix,
                        log_tensorboard=self.log_tensorboard,
                        log_params=self.log_params)

        return Q_nets

    def _loss(self, data, Q_nets):
        with tf.name_scope('loss'):
            target_Q = tf.stop_gradient(data['reward'] + self.gamma * Q_nets.target_next_Q, name='target_Q')
            Q_error = tf.abs(self.Q_nets - target_Q)
            
            priority = self._compute_priority(Q_error)

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            loss = tf.reduce_mean(data['IS_ratio'] * loss_func(Q_error))

        return priority, loss

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.variable_scope('loss', reuse=self.reuse):
                tf.summary.scalar('critic_loss_', self.loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q', tf.reduce_max(self.Q_nets.Q))
                tf.summary.scalar('min_Q', tf.reduce_min(self.Q_nets.Q))
