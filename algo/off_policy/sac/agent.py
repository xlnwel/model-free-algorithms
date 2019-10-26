import numpy as np
import tensorflow as tf

from algo.off_policy.basic_agent import OffPolicyOperation
from algo.off_policy.sac.networks import SoftPolicy, SoftQ, Temperature
from utility.losses import huber_loss
from utility.decorators import override
from utility.tf_utils import n_step_target, stats_summary
from utility.schedule import PiecewiseSchedule


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

        # learning rate schedule
        self.schedule_lr = 'schedule_lr' in args and args['schedule_lr']
        if self.schedule_lr:
            self.actor_lr_scheduler = PiecewiseSchedule([(0, 1e-4), (150000, 1e-4), (300000, 5e-5)], outside_value=5e-5)
            self.Q_lr_scheduler = PiecewiseSchedule([(0, 3e-4), (150000, 3e-4), (300000, 5e-5)], outside_value=5e-5)
            self.alpha_lr_scheduler = PiecewiseSchedule([(0, 1e-4), (150000, 1e-4), (300000, 5e-5)], outside_value=5e-5)
            
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

    @override(OffPolicyOperation)
    def _build_graph(self):
        if 'gpu' in self.device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)

        self.actor = self._actor()
        
        self._action_surrogate()

        self.critic = self._critic()

        if self.raw_temperature == 'auto':
            self.temperature = self._auto_temperature()
            self.alpha = self.temperature.alpha
            self.next_alpha = self.temperature.next_alpha
        else:
            # reward scaling indirectly affects the policy temperature
            # we neutralize the effect by scaling the temperature here
            # see my blog for more info https://xlnwel.github.io/blog/reinforcement%20learning/SAC/
            self.alpha = self.raw_temperature * self.buffer.reward_scale
            self.next_alpha = self.alpha

        self._compute_loss()
        self._optimize()
        
        self._log_loss()

    def _actor(self):
        policy_args = self.args['Policy']
        policy_args['max_action_repetitions'] = self.max_action_repetitions
        policy_args['polyak'] = self.args['polyak']
        return SoftPolicy('SoftPolicy',
                            policy_args,
                            self.graph,
                            self.data['state'],
                            self.data['next_state'],
                            self.action_dim,
                            scope_prefix=self.name,
                            log_tensorboard=self.log_tensorboard,
                            log_params=self.log_params)

    def _action_surrogate(self):
        self.action = self.actor.action
        self.action_det = self.actor.action_det
        self.next_action = self.actor.next_action
        self.logpi = self.actor.logpi
        self.next_logpi = self.actor.next_logpi
        
    def _critic(self):
        q_args = self.args['Q']
        q_args['polyak'] = self.args['polyak']
        return SoftQ('SoftQ',
                    q_args,
                    self.graph,
                    self.data['state'],
                    self.data['next_state'],
                    self.data['action'], 
                    self.action,
                    self.next_action,
                    scope_prefix=self.name,
                    log_tensorboard=self.log_tensorboard,
                    log_params=self.log_params)

    def _auto_temperature(self):
        return Temperature('Temperature',
                            self.args['Temperature'],
                            self.graph,
                            self.data['state'],
                            self.data['next_state'],
                            self.action,
                            self.next_action,
                            scope_prefix=self.name,
                            log_tensorboard=self.log_tensorboard,
                            log_params=self.log_params)

    def _compute_loss(self):
        with tf.name_scope('loss'):
            if self.raw_temperature == 'auto':
                self.alpha_loss = self._alpha_loss()
                self.loss = self.alpha_loss
            else:
                self.loss = 0
            self.actor_loss = self._actor_loss()
            self.priority, self.Q1_loss, self.Q2_loss, self.critic_loss = self._critic_loss()
            self.loss += self.actor_loss + self.critic_loss

    def _alpha_loss(self):
        target_entropy = -self.action_dim
        with tf.name_scope('alpha_loss'):
            return -tf.reduce_mean(self.data['IS_ratio'] * self.temperature.log_alpha 
                                * tf.stop_gradient(self.logpi + target_entropy))

    def _actor_loss(self):
        with tf.name_scope('actor_loss'):
            return tf.reduce_mean(self.data['IS_ratio'] 
                                * (self.alpha * self.logpi - self.critic.Q1_with_actor))

    def _critic_loss(self):
        with tf.name_scope('critic_loss'):
            n_V = tf.subtract(self.critic.next_Q_with_actor, self.next_alpha * self.next_logpi, name='n_V')
            target_Q = n_step_target(self.data['reward'], self.data['done'], 
                                    n_V, self.gamma, self.data['steps'])
            Q1_error = tf.abs(target_Q - self.critic.Q1, name='Q1_error')
            Q2_error = tf.abs(target_Q - self.critic.Q2, name='Q2_error')

            Q1_loss = tf.reduce_mean(self.data['IS_ratio'] * .5 * Q1_error**2)
            Q2_loss = tf.reduce_mean(self.data['IS_ratio'] * .5 * Q2_error**2)
            critic_loss = Q1_loss + Q2_loss

        priority = self._compute_priority((Q1_error + Q2_error) / 2.)

        return priority, Q1_loss, Q2_loss, critic_loss
    
    def _optimize(self):
        with tf.name_scope('optimizer'):
            opt_ops = []
            if self.raw_temperature == 'auto':
                _, self.alpha_lr, _, _, temp_op = self.temperature._optimization_op(self.alpha_loss, schedule_lr=self.schedule_lr)
                opt_ops.append(temp_op)
            _, self.actor_lr, self.opt_step, _, actor_opt_op = self.actor._optimization_op(self.actor_loss, opt_step=True, schedule_lr=self.schedule_lr)
            _, self.Q_lr, _, _, Q_opt_op = self.critic._optimization_op(self.critic_loss, schedule_lr=self.schedule_lr)
            opt_ops += [actor_opt_op, Q_opt_op]
            self.opt_op = tf.group(*opt_ops)

    @override(OffPolicyOperation)
    def _initialize_target_net(self):
        self.sess.run(self.actor.init_target_op + self.critic.init_target_op)

    @override(OffPolicyOperation)
    def _update_target_net(self):
        self.sess.run(self.actor.update_target_op + self.critic.update_target_op)

    @override(OffPolicyOperation)
    def _get_feeddict(self, t):
        return {
            self.actor_lr: self.actor_lr_scheduler.value(t),
            self.Q_lr: self.Q_lr_scheduler.value(t),
            self.alpha_lr: self.alpha_lr_scheduler.value(t)
        }

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('info'):
                stats_summary('reward', self.data['reward'], min=True, max=True, hist=True)
                with tf.name_scope('actor'):
                    stats_summary('logpi', self.actor.logpi)
                    tf.compat.v1.summary.scalar('actor_loss_', self.actor_loss)
                with tf.name_scope('critic'):
                    stats_summary('Q1_with_actor', self.critic.Q1_with_actor, min=True, max=True)
                    stats_summary('Q2_with_actor', self.critic.Q2_with_actor, min=True, max=True)
                    if self.buffer_type == 'proportional':
                        stats_summary('priority', self.priority, std=True, max=True, hist=True)
                    tf.compat.v1.summary.scalar('Q1_loss_', self.Q1_loss)
                    tf.compat.v1.summary.scalar('Q2_loss_', self.Q2_loss)
                    tf.compat.v1.summary.scalar('critic_loss_', self.critic_loss)
                    tf.compat.v1.summary.scalar('critic_loss_', self.critic_loss)
                if self.raw_temperature == 'auto':
                    with tf.name_scope('alpha'):
                        stats_summary('alpha', self.alpha, std=True)
                        tf.compat.v1.summary.scalar('alpha_loss', self.alpha_loss)
