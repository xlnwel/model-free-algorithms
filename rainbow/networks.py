import tensorflow as tf

from basic_model.basic_nets import Base

class QNets(Base):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 data,
                 num_actions, 
                 reuse=False, 
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
                         reuse=reuse, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params)

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


    def _build_plain_net(self, name, reuse):
        Q = self._plain_critic(self.state, self.action, self._reuse)
        Q_with_actor = self._plain_critic(self.state, self.actor_action, True)
    
        return Q, Q_with_actor

    def _build_distributional_net(self, name=None):
        def compute_Q(Z_support, Z_probs):
            return tf.reduce_sum(Z_support[None, :] * Z_probs, axis=1, keepdims=True)
            
        def distributional_ciritc(state, action, actor_action, Z_support, reuse):
            Z_probs = self._distributional_critic(state, action, reuse)
            Z_probs_with_actor = self._distributional_critic(state, actor_action, True)
            Q = compute_Q(Z_support, Z_probs)
            Q_with_actor = compute_Q(Z_support, Z_probs_with_actor)

            return Z_probs, Z_probs_with_actor, Q, Q_with_actor

        if name:
            with tf.variable_scope(name, reuse=self._reuse):
                Z_probs, Z_probs_with_actor, Q, Q_with_actor = distributional_ciritc(self.state, self.action, self.actor_action, self.Z_support, self._reuse)
        else:
            Z_probs, Z_probs_with_actor, Q, Q_with_actor = distributional_ciritc(self.state, self.action, self.actor_action, self.Z_support, self._reuse)

        return Z_probs, Z_probs_with_actor, Q, Q_with_actor

    def _conv_net(self, state, reuse=None):
        assert state.shape.as_list()[1:] == [84, 84, 1], f'Input image should be of shape (84, 84, 1), but get {state.shape.as_list()[1:]}'
        x = state

        with tf.variable_scope('convnet', reuse=reuse):
            x = self.conv_norm_activation(x, 32, 8, 4, padding='same', normalization=None)      # (21, 21, 32)
            x = self.conv_norm_activation(x, 64, 4, 2, padding='same', normalization=None)      # (10, 10, 64)
            x = self.conv_norm_activation(x, 64, 3, 1, padding='same', normalization=None)      # (10, 10, 64)
            x = tf.layers.flatten(x)

        return x

    def _dense_net(self, x, out_dim, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            x = self.dense_norm_activation(x, 512, normalization=None)
            x = self.dense(x, out_dim)

        return x