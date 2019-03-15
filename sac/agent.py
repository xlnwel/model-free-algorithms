import tensorflow as tf

from basic_model.off_policy import OffPolicy
from sac.networks import SoftPolicy, SoftV, SoftQ
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
                 log_tensorboard=True, 
                 log_params=False, 
                 log_score=True, 
                 device=None):
        # optional improvements
        options = args['options']
        self.n_steps = options['n_steps']
        self.critic_loss_type = options['loss_type']
        self.extra_critic_updates = options['extra_critic_updates']
        self.priority_type = options['priority']

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

        self.actor, self.V_nets, self.Q_nets = self._create_nets(self.data)
        
        self.action = self.actor.action
        self.logpi = -self.actor.action_distribution.neglogp(self.data['action'])

        self.priority, self.actor_loss, self.V_loss, self.Q_loss = self._loss(self.actor, self.V_nets, self.Q_nets, self.logpi)
        self.loss = self.actor_loss + self.V_loss + self.Q_loss

        self.actor_opt_op, self.global_step = self.actor._optimization_op(self.actor_loss, global_step=True)
        self.V_opt_op, _ = self.V_nets._optimization_op(self.V_loss)
        self.Q_opt_op, _ = self.Q_nets._optimization_op(self.Q_loss)
        self.critic_opt_op = tf.group(self.V_opt_op, self.Q_opt_op)

        self._log_loss()

    def _create_nets(self, data):
        scope_prefix = self.name
        actor = SoftPolicy('SoftPolicy', 
                            self.args['policy'],
                            self.graph,
                            data['state'],
                            self.env,
                            reuse=self.reuse, 
                            scope_prefix=scope_prefix,
                            log_tensorboard=self.log_tensorboard,
                            log_params=self.log_params)

        Vs = SoftV('SoftV',
                    self.args['V'],
                    self.graph,
                    data['state'],
                    data['next_state'],
                    reuse=self._reuse, 
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
                    reuse=self.reuse, 
                    scope_prefix=scope_prefix,
                    log_tensorboard=self.log_tensorboard,
                    log_params=self.log_params)

        return actor, Vs, Qs

    def _loss(self, policy, Vs, Qs, logpi):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                actor_loss = tf.reduce_mean(logpi - Qs.Q1_with_actor)

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            with tf.name_scope('V_loss'):
                target_V = tf.stop_gradient(Qs.Q_with_actor - logpi, name='target_V')
                TD_error = tf.abs(target_V - Vs.V)
                V_loss = .5 * loss_func(TD_error)

            with tf.name_scope('Q_loss'):
                target_Q = self._n_step_target(Vs.target_V)
                Q1_error = tf.abs(target_Q - Qs.Q1)
                Q2_error = tf.abs(target_Q - Qs.Q2)

                Q1_loss = .5 * loss_func(Q1_error)
                Q2_loss = .5 * loss_func(Q2_error)
                Q_loss = Q1_loss + Q2_loss

            priority = TD_error if self.priority_type == 'V' else (Q1_error + Q2_error) / 2.
            priority = self._compute_priority(priority)

        return priority, actor_loss, V_loss, Q_loss

    def _initialize_target_net(self):
        self.sess.run(self.V_nets.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.V_nets.update_target_op)

    def _log_loss(self):
        if self._log_tensorboard:
            with tf.name_scope('priority'):
                tf.summary.histogram('priority_', self.priority)
                tf.summary.scalar('priority_', tf.reduce_mean(self.priority))

            with tf.variable_scope('loss', reuse=self._reuse):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('V_loss_', self.V_loss)
                tf.summary.scalar('Q_loss_', self.Q_loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.Q_nets.Q_with_actor))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.Q_nets.Q_with_actor))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.Q_nets.Q_with_actor))

            with tf.name_scope('V'):
                tf.summary.scalar('V_', self.V_nets.V)
                tf.summary.scalar('target_V_', self.V_nets.target_V)
