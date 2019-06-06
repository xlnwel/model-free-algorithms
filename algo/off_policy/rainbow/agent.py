import tensorflow as tf

from algo.off_policy.basic_agent import OffPolicyOperation
from algo.off_policy.rainbow.networks import Networks
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
                 save=True, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_score=False, 
                 device=None):
        # optional improvements
        self.n_steps = args['n_steps']
        self.critic_loss_type = args['loss_type']
        self.update_rate = args['update_rate']

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

    def act(self, obs, return_q=False):
        obs = self.buffer.encode_recent_obs(obs)
        return super().act(obs, return_q)

    def learn(self):
        if self.update_step % self.update_rate == 0:
            super().learn()
        else:
            pass # do nothing

    def _build_graph(self):
        if 'gpu' in self.device:
            with tf.device('/cpu: 0'):
                self.data = self._prepare_data(self.buffer)
        else:
            self.data = self._prepare_data(self.buffer)

        self.nets = self._create_nets(self.data)
        self.action = self.nets.action

        self.priority, self.loss = self._loss(self.data, self.nets)

        self.opt_op, self.opt_step = self.nets._optimization_op(self.loss, opt_step=True)

        self._log_loss()

    def _create_nets(self, data):
        scope_prefix = self.name

        nets = Networks('Nets', 
                        self.args['networks'], 
                        self.graph, 
                        self.data, 
                        self.action_dim,
                        scope_prefix=scope_prefix,
                        log_tensorboard=self.log_tensorboard,
                        log_params=self.log_params)

        return nets

    def _loss(self, data, nets):
        with tf.name_scope('loss'):
            target_Q = n_step_target(self.data['reward'], self.data['done'],
                                    nets.target_next_Q, self.gamma, self.data['steps'])
            Q_error = tf.abs(nets.Q - target_Q)
            
            priority = self._compute_priority(Q_error)

            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            loss = tf.reduce_mean(data['IS_ratio'] * loss_func(Q_error))

        return priority, loss

    def _initialize_target_net(self):
        self.sess.run(self.nets.init_target_op)

    def _update_target_net(self):
        self.sess.run(self.nets.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('loss_', self.loss)
            
            with tf.name_scope('networks'):
                stats_summary(self.nets.Q, 'Q')
