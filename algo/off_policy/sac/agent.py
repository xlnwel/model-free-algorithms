import tensorflow as tf

from basic_model.off_policy import OffPolicy
from algo.off_policy.sac.networks import SoftPolicy, SoftV, SoftQ
from utility.losses import huber_loss


class Agent(OffPolicy):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=True, 
                 log_tensorboard=True, 
                 log_params=False, 
                 log_score=True, 
                 device=None):
        self.temperature = args['temperature']

        self.n_steps = args['n_steps']
        self.critic_loss_type = args['loss_type']
        self.extra_critic_updates = args['extra_critic_updates']
        self.priority_type = args['priority']
        self.reparameterize = args['reparameterize']

        super().__init__(name,
                         args,
                         env_args,
                         buffer_args,
                         sess_config=sess_config,
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
        
        self.action = tf.stop_gradient(self.actor.action) if self.reparameterize else self.actor.action
        self.logpi = -self.actor.action_distribution.neglogp(self.data['action'])

        self.priority, self.actor_loss, self.V_loss, self.Q_loss = self._loss(self.actor, self.V_nets, self.Q_nets, self.logpi)
        self.loss = self.actor_loss + self.V_loss + self.Q_loss

        self.actor_opt_op, self.opt_step = self.actor._optimization_op(self.actor_loss, opt_step=True)
        self.V_opt_op, _ = self.V_nets._optimization_op(self.V_loss)
        self.Q_opt_op, _ = self.Q_nets._optimization_op(self.Q_loss)
        self.critic_opt_op = tf.group(self.V_opt_op, self.Q_opt_op)
        self.opt_op = tf.group(self.actor_opt_op, self.critic_opt_op)

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
                    self.action_space,
                    scope_prefix=scope_prefix,
                    log_tensorboard=self.log_tensorboard,
                    log_params=self.log_params)

        return actor, Vs, Qs

    def _loss(self, policy, Vs, Qs, logpi):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                if self.reparameterize:
                    actor_loss = tf.reduce_mean(self.temperature * logpi - Qs.Q1_with_actor)
                else:
                    kl = tf.stop_gradient(self.temperature * self.logpi - Qs.Q_with_actor + Vs.V)
                    actor_loss = tf.reduce_mean(kl * self.logpi)

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            with tf.name_scope('V_loss'):
                target_V = tf.stop_gradient(Qs.Q_with_actor - self.temperature * logpi, name='target_V')
                TD_error = tf.abs(target_V - Vs.V)
                V_loss = .5 * tf.reduce_mean(loss_func(TD_error))

            with tf.name_scope('Q_loss'):
                target_Q = self._n_step_target(Vs.target_V)
                Q1_error = tf.abs(target_Q - Qs.Q1)
                Q2_error = tf.abs(target_Q - Qs.Q2)

                Q1_loss = loss_func(Q1_error)
                Q2_loss = loss_func(Q2_error)
                Q_loss = .5 * tf.reduce_mean(Q1_loss + Q2_loss)

            priority = TD_error if self.priority_type == 'V' else (Q1_error + Q2_error) / 2.
            priority = self._compute_priority(priority)

        return priority, actor_loss, V_loss, Q_loss

    def _initialize_target_net(self):
        self.sess.run(self.V_nets.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.V_nets.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('priority'):
                tf.summary.histogram('priority_', self.priority)
                tf.summary.scalar('priority_', tf.reduce_mean(self.priority))

            with tf.name_scope('loss'):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('V_loss_', self.V_loss)
                tf.summary.scalar('Q_loss_', self.Q_loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.Q_nets.Q_with_actor))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.Q_nets.Q_with_actor))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.Q_nets.Q_with_actor))

            with tf.name_scope('V'):
                tf.summary.scalar('V_', tf.reduce_mean(self.V_nets.V))
                tf.summary.scalar('target_V_', tf.reduce_mean(self.V_nets.target_V))
