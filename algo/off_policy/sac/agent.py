import tensorflow as tf

from algo.off_policy.basic_agent import OffPolicyOperation
from algo.off_policy.sac.networks import SoftPolicy, SoftQ, Temperature
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
                 log=False,
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        self.raw_temperature = args['temperature']
        self.critic_loss_type = args['loss_type']

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

    def _build_graph(self):
        if 'gpu' in self.device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)

        self.actor, self.Q_nets = self._create_nets(self.data)
        
        self.action_det = self.actor.action_det
        self.action = self.actor.action
        self.logpi = self.actor.logpi

        opt_ops = []
        if self.raw_temperature == 'auto':
            self.temperature = Temperature('Temperature',
                                            self.args['Temperature'],
                                            self.graph,
                                            self.data['state'],
                                            self.data['next_state'],
                                            self.actor,
                                            scope_prefix=self.name,
                                            log_tensorboard=self.log_tensorboard,
                                            log_params=self.log_params)
            self.alpha = self.temperature.alpha
            self.next_alpha = self.temperature.next_alpha
            target_entropy = -self.action_dim
            self.alpha_loss = self._alpha_loss(self.temperature.log_alpha, self.logpi, target_entropy)
            _, _, _, _, temp_op = self.temperature._optimization_op(self.alpha_loss)
            opt_ops.append(temp_op)
        else:
            # reward scaling indirectly affects the policy temperature
            # we neutralize the effect by scaling the temperature here
            # see my blog for more info https://xlnwel.github.io/blog/reinforcement%20learning/SAC/
            self.alpha = self.raw_temperature * self.buffer.reward_scale
            self.next_alpha = self.alpha

        self.priority, losses = self._loss(self.actor, self.Q_nets, self.logpi)
        self.actor_loss, self.Q_loss, self.loss = losses

        _, _, self.opt_step, _, actor_opt_op = self.actor._optimization_op(self.actor_loss, opt_step=True)
        _, _, _, _, Q_opt_op = self.Q_nets._optimization_op(self.Q_loss)
        opt_ops += [actor_opt_op, Q_opt_op]
        self.opt_op = tf.group(*opt_ops)

        self._log_loss()

    def _create_nets(self, data):
        scope_prefix = self.name
        actor = SoftPolicy('SoftPolicy', 
                            self.args['Policy'],
                            self.graph,
                            data['state'],
                            data['next_state'],
                            self.env,
                            scope_prefix=scope_prefix,
                            log_tensorboard=self.log_tensorboard,
                            log_params=self.log_params)

        Qs = SoftQ('SoftQ',
                    self.args['Q'],
                    self.graph,
                    data['state'],
                    data['next_state'],
                    data['action'], 
                    actor,
                    self.action_dim,
                    scope_prefix=scope_prefix,
                    log_tensorboard=self.log_tensorboard,
                    log_params=self.log_params)

        return actor, Qs

    def _loss(self, policy, Qs, logpi):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                actor_loss = tf.reduce_mean(self.data['IS_ratio'] * (self.alpha * logpi - Qs.Q1_with_actor))

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square

            with tf.name_scope('Q_loss'):
                steps = self.data['steps']
                target_Q = n_step_target(self.data['reward'], self.data['done'], 
                                         Qs.next_Q_with_actor - self.next_alpha * policy.next_logpi, 
                                         self.gamma)
                Q1_error = tf.abs(target_Q - Qs.Q1)
                Q2_error = tf.abs(target_Q - Qs.Q2)

                Q1_loss = tf.reduce_mean(self.data['IS_ratio'] * (loss_func(Q1_error)))
                Q2_loss = tf.reduce_mean(self.data['IS_ratio'] * (loss_func(Q2_error)))
                Q_loss = Q1_loss + Q2_loss

            loss = actor_loss + Q_loss

        priority = (Q1_error + Q2_error) / 2.
        priority = self._compute_priority(priority)

        return priority, (actor_loss, Q_loss, loss)
    
    def _alpha_loss(self, log_alpha, logpi, target_entropy):
        with tf.name_scope('alpha_loss'):
            return -tf.reduce_mean(log_alpha * tf.stop_gradient(logpi + target_entropy))

    def _initialize_target_net(self):
        self.sess.run(self.Q_nets.init_target_op)
        if self.actor.has_target_net:
            self.sess.run(self.actor.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.Q_nets.update_target_op)
        if self.actor.has_target_net:
            self.sess.run(self.actor.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('Q_loss_', self.Q_loss)
            if self.raw_temperature == 'auto':
                with tf.name_scope('temperature'):
                    stats_summary('alpha', self.alpha)
                    stats_summary('alpha_loss', self.alpha_loss)
