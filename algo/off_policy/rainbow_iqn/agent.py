import tensorflow as tf

from algo.off_policy.basic_agent import OffPolicyOperation
from algo.off_policy.rainbow_iqn.networks import Networks
from utility.losses import huber_loss
from utility.display import pwc
from utility.tf_utils import n_step_target, stats_summary


class Agent(OffPolicyOperation):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=False, 
                 log=False,
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        # optional improvements
        self.n_steps = args['n_steps']
        self.critic_loss_type = args['loss_type']
        self.polyak = args['polyak'] if 'polyak' in args else .995
        self.algo = args['Qnets']['algo']

        super().__init__(name,
                         args,
                         env_args,
                         buffer_args,
                         sess_config=sess_config,
                         save=save,
                         log=log,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_stats=log_stats,
                         device=device)

    """ Implementation """
    def _build_graph(self):
        self.data = self._prepare_data(self.buffer)

        self.Qnets = self._create_nets()
        self.action = self.Qnets.best_action

        self.priority, self.loss = self._loss()

        _, _, self.opt_step, _, self.opt_op = self.Qnets._optimization_op(self.loss, 
                                                                          tvars=self.Qnets.main_variables,
                                                                          opt_step=True)

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_loss()

    def _create_nets(self):
        scope_prefix = self.name
        self.args['Qnets']['batch_size'] = self.args['batch_size']
        Qnets = Networks('Nets', 
                        self.args['Qnets'], 
                        self.graph, 
                        self.data, 
                        self.action_dim,    # n_actions
                        scope_prefix=scope_prefix,
                        log_tensorboard=self.log_tensorboard,
                        log_params=self.log_params)

        return Qnets

    def _loss(self):
        if self.algo == 'iqn':
            return self._iqn_loss()
        else:
            return self._q_loss()

    def _iqn_loss(self):
        def tiled_n_step_target():
            n_quantiles = self.args['Qnets']['N_prime']
            
            reward_tiled = tf.reshape(tf.tile(self.data['reward'], [n_quantiles, 1]),
                                      [n_quantiles, -1, 1])
            done_tiled = tf.reshape(tf.tile(self.data['done'], [n_quantiles, 1]),
                                      [n_quantiles, -1, 1])
            steps_tiled = tf.reshape(tf.tile(self.data['steps'], [n_quantiles, 1]),
                                      [n_quantiles, -1, 1])
            return n_step_target(reward_tiled, done_tiled, 
                                 self.Qnets.quantile_values_next_target,
                                 self.gamma, steps_tiled)

        def quantile_regression_loss(u):
            abs_part = tf.abs(self.Qnets.quantiles - tf.where(u < 0, tf.ones_like(u), tf.zeros_like(u)))
            huber = huber_loss(u, delta=self.args['Qnets']['delta'])
            
            qr_loss = tf.reduce_sum(tf.reduce_mean(abs_part * huber, axis=2), axis=1)   # [B]
            loss = tf.reduce_mean(qr_loss)

            return loss

        with tf.name_scope('priority'):
            Q_target = n_step_target(self.data['reward'], self.data['done'],
                                    self.Qnets.Q_next_target, self.gamma, self.data['steps'])
            Q_error = tf.abs(self.Qnets.Q - Q_target)
            priority = self._compute_priority(Q_error)

        with tf.name_scope('loss'):
            quantile_values_target = tiled_n_step_target()
            quantile_values_target = tf.transpose(quantile_values_target, [1, 2, 0])    # [B, 1, N']
            quantile_values = tf.transpose(self.Qnets.quantile_values, [1, 0, 2])       # [B, N, 1]
            quantile_error = tf.abs(quantile_values - quantile_values_target)

            loss = quantile_regression_loss(quantile_error)

        return priority, loss

    def _q_loss(self):
        with tf.name_scope('loss'):
            Q_target = n_step_target(self.data['reward'], self.data['done'],
                                    self.Qnets.Q_next_target, self.gamma, self.data['steps'])
            Q_error = tf.abs(self.Qnets.Q - Q_target)
            
            priority = self._compute_priority(Q_error)

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            loss = tf.reduce_mean(self.data['IS_ratio'] * loss_func(Q_error), name='loss')

        return priority, loss

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.Qnets.target_variables, self.Qnets.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.polyak * v[0] + (1. - self.polyak) * v[1], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

    def _initialize_target_net(self):
        self.sess.run(self.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.compat.v1.summary.scalar('loss_', self.loss)
            
            with tf.name_scope('networks'):
                stats_summary('Q', self.Qnets.Q)

