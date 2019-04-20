import tensorflow as tf

from basic_model.basic_nets import Base
from utility.utils import assert_colorize

class QNets(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 data,
                 num_actions, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.action = data['action']
        self.next_state = data['next_state']
        self.reward = data['reward']
        self.num_actions = num_actions
        super().__init__(name, 
                         args, 
                         graph, 
                         data['state'],
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

    @property
    def main_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/main')
    
    @property
    def target_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/target')

    """ Implementation """
    def _build_graph(self):
        x = self._conv_net(self.state)
        next_x = self._conv_net(self.next_state)
        self.Q_dist = self._dense_net(x, self.num_actions, reuse=False, name='main')
        self.next_Q_dist = self._dense_net(next_x, self.num_actions, reuse=True, name='main')
        self.target_Q_dist = self._dense_net(next_x, self.num_actions, reuse=False, name='target')

        self.Q = tf.reduce_sum(tf.one_hot(self.action, self.num_actions) * self.Q_dist, axis=1)
        self.action = tf.argmax(self.Q_dist, axis=1)
        self.next_action = tf.argmax(self.next_Q_dist, axis=1)
        self.target_next_Q = tf.reduce_sum(tf.one_hot(self.next_action, self.num_actions) * self.target_Q_dist, axis=1)

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

    # def _build_distributional_net(self, name=None):
    #     def compute_Q(Z_support, Z_probs):
    #         return tf.reduce_sum(Z_support[None, :] * Z_probs, axis=1, keepdims=True)
            
    #     def distributional_ciritc(state, action, actor_action, Z_support):
    #         Z_probs = self._distributional_critic(state, action)
    #         Z_probs_with_actor = self._distributional_critic(state, actor_action, True)
    #         Q = compute_Q(Z_support, Z_probs)
    #         Q_with_actor = compute_Q(Z_support, Z_probs_with_actor)

    #         return Z_probs, Z_probs_with_actor, Q, Q_with_actor

    #     if name:
    #         with tf.variable_scope(name):
    #             Z_probs, Z_probs_with_actor, Q, Q_with_actor = distributional_ciritc(self.state, self.action, self.actor_action, self.Z_support, self._reuse)
    #     else:
    #         Z_probs, Z_probs_with_actor, Q, Q_with_actor = distributional_ciritc(self.state, self.action, self.actor_action, self.Z_support, self._reuse)

    #     return Z_probs, Z_probs_with_actor, Q, Q_with_actor

    def _conv_net(self, state):
        assert_colorize(state.shape.as_list()[1:] == [84, 84, 1], f'Input image should be of shape (84, 84, 1), but get {state.shape.as_list()[1:]}')
        x = state

        with tf.variable_scope('convnet'):
            x = self.conv_norm_activation(x, 32, 8, 4, padding='same', normalization=None)      # (21, 21, 32)
            x = self.conv_norm_activation(x, 64, 4, 2, padding='same', normalization=None)      # (10, 10, 64)
            x = self.conv_norm_activation(x, 64, 3, 1, padding='same', normalization=None)      # (10, 10, 64)
            x = tf.layers.flatten(x)

        return x

    def _dense_net(self, x, out_dim, name):
        with tf.variable_scope(name):
            x = self.noisy_norm_activation(x, 512, normalization=None)
            x = self.noisy(x, out_dim)

        return x

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op