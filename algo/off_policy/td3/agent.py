import tensorflow as tf

from algo.off_policy.basic_agent import OffPolicyOperation
from algo.off_policy.td3.networks import Actor, Critic, DoubleCritic
from utility.losses import huber_loss
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
        self.critic_loss_type = args['critic']['loss_type']
        self.polyak = args['polyak'] if 'polyak' in args else .995
        
        # learning rate schedule
        self.schedule_lr = 'schedule_lr' in args and args['schedule_lr']
        if self.schedule_lr:
            self.actor_lr_scheduler = PiecewiseSchedule([(0, 1e-4), (150000, 1e-4), (300000, 5e-5)], outside_value=5e-5)
            self.critic_lr_scheduler = PiecewiseSchedule([(0, 3e-4), (150000, 3e-4), (300000, 5e-5)], outside_value=5e-5)

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

    @property
    def main_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    @property
    def target_variables(self):
        return self.target_actor.trainable_variables + self.target_critic.trainable_variables

    """ Implementation """
    def _build_graph(self):
        if self.device and 'GPU' in self.device:
            with tf.device('/CPU: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)
            
        self.actor, self.critic, self.target_actor, self.target_critic = self._create_main_target_actor_critic()
        self.action_det_repr = self.action_repr = self.actor.action

        self.priority, self.actor_loss, self.critic_loss = self._loss()
        self.loss = self.actor_loss + self.critic_loss
    
        _, self.actor_lr, self.opt_step, _, self.actor_opt_op = self.actor._optimization_op(self.actor_loss, opt_step=True, schedule_lr=self.schedule_lr)
        _, self.critic_lr, _, _, self.critic_opt_op = self.critic._optimization_op(self.critic_loss, schedule_lr=self.schedule_lr)
        self.opt_op = tf.group(self.actor_opt_op, self.critic_opt_op)

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_loss()

    def _create_main_target_actor_critic(self):
        # main actor-critic
        actor, critic = self._create_actor_critic(is_target=False)
        # target actor-critic
        target_actor, target_critic = self._create_actor_critic(is_target=True)

        return actor, critic, target_actor, target_critic
        
    def _create_actor_critic(self, is_target):
        log_tensorboard = False if is_target else self.log_tensorboard
        log_params = False if is_target else self.log_params

        scope_name = 'target' if is_target else 'main'
        state = self.data['next_state'] if is_target else self.data['state']
        scope_prefix = self.name + '/' + scope_name
        self.args['actor']['max_action_repetitions'] = self.max_action_repetitions
        
        with tf.variable_scope(scope_name):
            actor = Actor('actor', 
                          self.args['actor'], 
                          self.graph,
                          state, 
                          self.action_dim, 
                          scope_prefix=scope_prefix, 
                          log_tensorboard=log_tensorboard, 
                          log_params=log_params)

            critic = DoubleCritic('critic', 
                                 self.args['critic'],
                                 self.graph,
                                 state,
                                 self.data['action'], 
                                 actor.action,
                                 self.action_dim,
                                 scope_prefix=scope_prefix, 
                                 log_tensorboard=log_tensorboard,
                                 log_params=log_params)
        
        return actor, critic

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                # importance sampling draw down performance
                actor_loss = -tf.reduce_mean(self.data['IS_ratio'] * self.critic.Q1_with_actor)
                # actor_loss = -tf.reduce_mean(self.critic.Q1_with_actor)

            with tf.name_scope('critic_loss'):
                target_Q = n_step_target(self.data['reward'], self.data['done'], 
                                         self.target_critic.Q_with_actor, 
                                         self.gamma, self.data['steps'])
                
                TD_error1 = tf.abs(target_Q - self.critic.Q1, name='TD_error1')
                TD_error2 = tf.abs(target_Q - self.critic.Q2, name='TD_error2')
                
                priority = self._compute_priority((TD_error1 + TD_error2) / 2.)

                loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
                TD_squared = (loss_func(TD_error1) 
                              + loss_func(TD_error2))

                critic_loss = tf.reduce_mean(self.data['IS_ratio'] * TD_squared)

        return priority, actor_loss, critic_loss

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
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
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('critic_loss_', self.critic_loss)

            with tf.name_scope('info'):
                stats_summary('Q_with_actor', self.critic.Q_with_actor, max=True, hist=True)
                stats_summary('reward', self.data['reward'], min=True, hist=True)
                stats_summary('priority', self.priority, hist=True, max=True)

    def _get_feeddict(self, t):
        return {
            self.actor_lr: self.actor_lr_scheduler.value(t),
            self.critic_lr: self.critic_lr_scheduler.value(t)
        }
