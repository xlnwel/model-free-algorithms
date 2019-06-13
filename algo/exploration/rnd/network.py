import numpy as np
import tensorflow as tf

from basic_model.basic_nets import Base
from utility.run_avg import RunningMeanStd


class RND(Base):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 next_state_ph,
                 scope_prefix,
                 log_tensorboard=False,
                 log_params=False):
        self.next_state_ph = next_state_ph
        self.gamma = args['intrinsic_gamma']
        self.returns = 0
        self.runavg_returns = RunningMeanStd()

        super().__init__(name, args, graph, log_tensorboard=log_tensorboard, log_params=log_params)

    def compute_intrinsic_rewards(self, sess, next_state, normalize=False):
        rewards = sess.run(self.reward, feed_dict={self.next_state_ph: next_state})
        if normalize:
            self.update_runavg_returns(rewards)
            rewards = self.normalize_rewards(rewards)
        
        return rewards

    def update_runavg_returns(self, rewards):
        def return_update(rewards):
            self.returns = self.gamma * self.returns + rewards
            return self.returns

        int_returns = np.array([return_update(r) for r in rewards.T[::-1]])
        self.runavg_returns.update(int_returns.ravel())

    def normalize_rewards(self, rewards):
        return rewards / np.sqrt(self.runavg_returns.var)

    def _build_graph(self):
        self.target_repr = self._random_net(self.next_state_ph, self.args['units'], self.args['repr_dim'], self.args['norm'], 'target')
        self.main_repr = self._random_net(self.next_state_ph, self.args['units'], self.args['repr_dim'], self.args['norm'], 'main')

        self.reward = self._reward(self.target_repr, self.main_repr)
        self.loss = self._loss(self.target_repr, self.main_repr)

        self.opt_op, self.opt_step = self._optimization_op(self.loss, opt_step=True)

    def _random_net(self, next_state_ph, units, out_dim, norm, name):
        x = next_state_ph
        with tf.variable_scope(name):
            for u in enumerate(units):
                # TODO: uniform weight initialization
                x = self.dense_norm_activation(x, u, norm)

            x = self.dense(x, out_dim, name='repr')

        return x

    def _reward(self, target_repr, main_repr):
        reward = tf.reduce_mean((tf.stop_gradient(target_repr) - main_repr)**2, axis=-1)

        return reward

    def _loss(self, target_repr, main_repr):
        loss = tf.reduce_mean((tf.stop_gradient(target_repr) - main_repr)**2, axis=-1)
        # only a fraction of experiences are used for predictor update 
        mask = tf.random.uniform(shape=tf.shape(loss))
        mask = tf.cast(mask <= self.args['proportion_of_exp_used_for_predictor_update'], tf.float32)
        # loss from the original implementation:
        # https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/policies/cnn_gru_policy_dynamics.py#L192
        loss = tf.reduce_sum(mask * loss) / tf.maximum(tf.reduce_sum(mask), 1.)

        return loss
