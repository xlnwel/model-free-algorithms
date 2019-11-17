import numpy as np
import tensorflow as tf
import gym

from basic_model.model import Module
from utility import tf_utils, tf_distributions
from utility.schedule import PiecewiseSchedule
from utility.display import pwc
from utility.rl_losses import ppo_loss, clipped_value_loss
from env.gym_env import action_dist_type


class ActorCritic(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 env_vec,
                 env_phs,
                 scope_prefix,
                 log_tensorboard=False,
                 log_params=False):
        self.env_vec = env_vec
        self.env_phs = env_phs
        self.clip_range = args['clip_range']
        self.use_lstm = args['use_lstm']
        self.batch_seq_len = args['batch_seq_len']

        super().__init__(name,
                         args,
                         graph,
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)
        
    """ Implementation """
    def _build_graph(self):
        x = self.env_phs['state']
        
        if self.args['common']:
            # actor and critic share initial networks
            x = self._common_dense(x, self.args['common_dense_units'])
            if self.use_lstm:
                x, self.initial_state, self.final_state = self._common_lstm(x, self.args['common_lstm_units'])
            actor_output = self._policy_head(x, self.args['actor_units'], self.env_vec.action_dim, 
                                            discrete=self.env_vec.is_action_discrete)
            self.V = self._V_head(x, self.args['critic_units'])
        else:
            # independent actor & critic networks
            actor_output = self._policy_net(x, self.args['actor_units'], 
                                            self.env_vec.action_dim, 
                                            discrete=self.env_vec.is_action_discrete)

            self.V = self._v_net(x, self.args['critic_units'])
            if self.use_lstm:
                self.initial_state = [*self.actor_init_state, *self.critic_init_state]
                self.final_state = [*self.actor_final_state, *self.critic_final_state]

        self.action_distribution = action_dist_type(self.env_vec)(actor_output)
        action = self.action_distribution.sample()
        self.logpi = self.action_distribution.logp(tf.stop_gradient(self.action))
        self.action = tf.tanh(self.action)
        sub = 2 * tf.reduce_sum(tf.log(2.) + action - tf.nn.softplus(2 * action), axis=1, keepdims=True)
        self.logpi -= sub
        # losses
        self.ppo_loss, self.policy_loss, self.V_loss, self.entropy, self.approx_kl, self.p_clip_frac, self.v_clip_frac = self._loss()

        # optimizer
        schedule_policy_lr = 'schedule_policy_lr' in self.args and self.args['schedule_policy_lr']
        schedule_value_lr = 'schedule_value_lr' in self.args and self.args['schedule_value_lr']
        policy_opt_info = self._optimization_op(self.policy_loss, 
                                                opt_step=True, 
                                                schedule_lr=schedule_policy_lr, 
                                                name='policy')
        _, self.policy_lr, self.opt_step, self.policy_grads_and_vars, self.policy_optop = policy_opt_info
        _, self.v_lr, _, self.v_grads_and_vars, self.v_optop = self._optimization_op(self.V_loss,
                                                                                     schedule_lr=schedule_value_lr,  
                                                                                     name='value')
        self.grads_and_vars = self.policy_grads_and_vars + self.v_grads_and_vars
        self.grads = [gv[0] for gv in self.grads_and_vars if gv[0] is not None]

    """ Code for shared policy and value network"""
    def _common_dense(self, x, units, name='common_dense'):
        with tf.variable_scope(name):
            x = self._feedforward_net(x, units)

        return x

    def _common_lstm(self, x, units, name='common_lstm'):
        # initial&final state list for multi-layer LSTM
        init_state_list, final_state_list = [], []
        with tf.variable_scope(name):
            u = x.shape.as_list()[-1]
            x = tf.reshape(x, (self.env_vec.n_envs, -1, u))
            for u in units:
                x, (init_state, final_state) = self.lstm(x, u, return_sequences=True)
                
                init_state_list += init_state
                final_state_list += final_state

            x = tf.reshape(x, (-1, u))

        return x, init_state_list, final_state_list

    def _policy_head(self, x, units, action_dim, discrete=False, name='policy_head'):
        output_name = 'action_logits' if discrete else 'action_mean'
        with tf.variable_scope(name):
            x = self._feedforward_net(x, units, action_dim, output_name)

            if discrete:
                return x
            else:
                logstd = tf.get_variable('action_logstd', [action_dim], tf.float32)
                
                return x, logstd

    def _V_head(self, x, units, name='V_head'):
        with tf.variable_scope(name):
            x = self._feedforward_net(x, units, 1, name)

        return x

    """ Code for separate policy and value network """
    def _policy_net(self, x, units, action_dim, discrete=False, name='policy_net'):
        with tf.variable_scope(name):
            x = self._common_dense(x, self.args['common_dense_units'], name='dense')
            if self.use_lstm:
                x, self.actor_init_state, self.actor_final_state = self._common_lstm(x, self.args['common_lstm_units'], name='lstm')

            output = self._policy_head(x, self.args['actor_units'], self.env_vec.action_dim, 
                                        discrete=self.env_vec.is_action_discrete)
            
            return output

    def _v_net(self, x, units, name='V_net'):
        with tf.variable_scope(name):
            x = self._common_dense(x, self.args['common_dense_units'], name='dense')
            if self.use_lstm:
                x, self.critic_init_state, self.critic_final_state = self._common_lstm(x, self.args['common_lstm_units'], name='lstm')

            x = self._V_head(x, self.args['critic_units'])

        return x

    def _feedforward_net(self, x, units, output_dim=None, output_name=None):
        for u in units:
            x = self.dense_norm_activation(x, u, norm=tf_utils.get_norm(self.args['norm']))
    
        if output_dim and output_name:
            x = self.dense(x, output_dim, name=output_name)

        return x

    """ Losses """
    def _loss(self):
        with tf.name_scope('loss'):
            n = tf.reduce_sum(self.env_phs['mask'], name='num_true_entries')   # the number of True entries in mask

            loss_info = ppo_loss(self.logpi, self.env_phs['old_logpi'], 
                                self.env_phs['advantage'], self.clip_range, 
                                self.action_distribution.entropy(), 
                                self.env_phs['mask'], n)
            pg_loss, entropy, approx_kl, p_clip_frac = loss_info
            V_loss, v_clip_frac = clipped_value_loss(self.V, self.env_phs['return'], 
                                                    self.env_phs['value'], self.clip_range, 
                                                    self.env_phs['mask'], n)
            V_loss = self.args['value_coef'] * V_loss
            policy_loss = pg_loss - self.env_phs['entropy_coef'] * entropy + self.args['kl_coef'] * approx_kl

        return pg_loss, policy_loss, V_loss, entropy, approx_kl, p_clip_frac, v_clip_frac
