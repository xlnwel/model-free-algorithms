import tensorflow as tf

from basic_model.off_policy import OffPolicy
from td3_rainbow.networks import Actor, Critic, DoubleCritic
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
        options = args['options']
        self.n_steps = options['n_steps']
        self.critic_loss_type = args['critic']['loss_type']
        self.extra_critic_updates = args['critic']['extra_updates']
        
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

    @property
    def main_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    @property
    def target_variables(self):
        return self._target_actor.trainable_variables + self._target_critic.trainable_variables

    """ Implementation """
    def _build_graph(self):
        if 'gpu' in self.device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)
            
        self.actor, self.critic, self._target_actor, self._target_critic = self._create_main_target_actor_critic()

        self.priority, self.actor_loss, self.critic_loss = self._loss()
        self.loss = self.actor_loss + self.critic_loss
    
        self.actor_opt_op, self.global_step = self.actor._optimization_op(self.actor_loss, global_step=True)
        self.critic_opt_op, _ = self.critic._optimization_op(self.critic_loss)

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
        
        with tf.variable_scope(scope_name, reuse=self.reuse):
            actor = Actor('actor', 
                          self.args['actor'], 
                          self.graph,
                          state, 
                          self.action_dim, 
                          reuse=self.reuse, 
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
                                 reuse=self.reuse, 
                                 scope_prefix=scope_prefix, 
                                 log_tensorboard=log_tensorboard,
                                 log_params=log_params)
        
        return actor, critic

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                actor_loss = -tf.reduce_mean(self.data['IS_ratio'] * self.critic.Q1_with_actor)

            with tf.name_scope('critic_loss'):
                target_Q = self._n_step_target(self._target_critic.Q_with_actor)
                
                TD_error1 = tf.abs(target_Q - self.critic.Q1, name='TD_error1')
                TD_error2 = tf.abs(target_Q - self.critic.Q2, name='TD_error2')
                with tf.name_scope(name='priority'):
                    priority = self._compute_priority((TD_error1 + TD_error2) / 2.)

                loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
                TD_squared = loss_func(TD_error1) + loss_func(TD_error2)

                critic_loss = tf.reduce_mean(self.data['IS_ratio'] * TD_squared)

        return priority, actor_loss, critic_loss

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

    def _initialize_target_net(self):
        self.sess.run(self.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.variable_scope('loss', reuse=self.reuse):
                tf.summary.scalar('actor_loss_', self.actor_loss)
                tf.summary.scalar('critic_loss_', self.critic_loss)
            
            with tf.name_scope('Q'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.critic.Q_with_actor))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.critic.Q_with_actor))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.critic.Q_with_actor))
            