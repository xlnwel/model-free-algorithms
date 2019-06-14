import tensorflow as tf

from algo.off_policy.basic_agent import OffPolicyOperation
from algo.off_policy.sac.networks import SoftPolicy, SoftV, SoftQ
from utility.losses import huber_loss
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
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        self.temperature = args['temperature']
        self.critic_loss_type = args['loss_type']
        self.priority_type = args['priority']

        super().__init__(name,
                         args,
                         env_args,
                         buffer_args,
                         sess_config=sess_config,
                         save=save,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_stats=log_stats,
                         device=device)

    def _build_graph(self):
        if 'gpu' in self.device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)

        self.actor, self.V_nets, self.Q_nets = self._create_nets(self.data)
        
        self.action = self.actor.action
        self.logpi = self.actor.logpi

        self.priority, losses = self._loss(self.actor, self.V_nets, self.Q_nets, self.logpi)
        self.actor_loss, self.V_loss, self.Q_loss, self.loss = losses

        actor_opt_op, self.opt_step = self.actor._optimization_op(self.actor_loss, opt_step=True)
        V_opt_op, _ = self.V_nets._optimization_op(self.V_loss)
        Q_opt_op, _ = self.Q_nets._optimization_op(self.Q_loss)
        self.opt_op = tf.group(actor_opt_op, V_opt_op, Q_opt_op)

        self._log_loss()

    def _create_nets(self, data):
        scope_prefix = self.name
        actor = SoftPolicy('SoftPolicy', 
                            self.args['policy'],
                            self.graph,
                            data['state'],
                            self.env,
                            scope_prefix=scope_prefix,
                            log_tensorboard=self.log_tensorboard,
                            log_params=self.log_params)

        Vs = SoftV('SoftV',
                    self.args['V'],
                    self.graph,
                    data['state'],
                    data['next_state'],
                    scope_prefix=scope_prefix,
                    log_tensorboard=self.log_tensorboard,
                    log_params=self.log_params)

        Qs = SoftQ('SoftQ',
                    self.args['Q'],
                    self.graph,
                    data['state'],
                    data['action'], 
                    actor.action,
                    self.action_dim,
                    scope_prefix=scope_prefix,
                    log_tensorboard=self.log_tensorboard,
                    log_params=self.log_params)

        return actor, Vs, Qs

    def _loss(self, policy, Vs, Qs, logpi):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                actor_loss = tf.reduce_mean(self.data['IS_ratio'] * (self.temperature * logpi - Qs.Q1_with_actor))

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            with tf.name_scope('V_loss'):
                target_V = tf.stop_gradient(Qs.Q_with_actor - self.temperature * logpi, name='target_V')
                TD_error = tf.abs(target_V - Vs.V)
                V_loss = tf.reduce_mean(self.data['IS_ratio'] * (loss_func(TD_error)))

            with tf.name_scope('Q_loss'):
                target_Q = n_step_target(self.data['reward'], self.data['done'], 
                                        Vs.V_next, self.gamma, self.data['steps'])
                Q1_error = tf.abs(target_Q - Qs.Q1)
                Q2_error = tf.abs(target_Q - Qs.Q2)

                Q1_loss = tf.reduce_mean(self.data['IS_ratio'] * (loss_func(Q1_error)))
                Q2_loss = tf.reduce_mean(self.data['IS_ratio'] * (loss_func(Q2_error)))
                Q_loss = Q1_loss + Q2_loss

            loss = actor_loss + V_loss + Q_loss

        priority = TD_error if self.priority_type == 'V' else (Q1_error + Q2_error) / 2.
        priority = self._compute_priority(priority)

        return priority, (actor_loss, V_loss, Q_loss, loss)

    def _initialize_target_net(self):
        self.sess.run(self.V_nets.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.V_nets.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('V_loss_', self.V_loss)
                tf.summary.scalar('Q_loss_', self.Q_loss)
                tf.summary.scalar('loss_', self.loss)
