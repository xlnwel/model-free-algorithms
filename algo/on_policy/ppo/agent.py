import os
import numpy as np
import tensorflow as tf
import ray

from env.gym_env import GymEnvVec
from utility.losses import huber_loss
from basic_model.model import Model
from algo.on_policy.ppo.networks import ActorCritic
from algo.on_policy.ppo.buffer import PPOBuffer


class Agent(Model):
    """ Interface """
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 save=True,
                 log_tensorboard=True,
                 log_params=False,
                 log_score=True,
                 device=None):
        # hyperparameters
        self.gamma = args['gamma']
        self.gae_discount = self.gamma * args['lambda']
        self.seq_len = args['seq_len']
        self.use_lstm = args['ac']['use_lstm']

        self.entropy_coef = args['ac']['entropy_coef']
        self.n_minibatches = args['n_minibatches']
        self.minibach_size = self.seq_len // self.n_minibatches
        self.minibatch_idx = 0

        # environment info
        self.env_vec = GymEnvVec(env_args['name'], args['n_envs'], 
                                 max_episode_steps=self.seq_len, 
                                 seed=np.random.randint(1000))
        super().__init__(name, args, 
                         sess_config=sess_config,
                         save=save, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_score=log_score,
                         device=device)

        self.buffer = PPOBuffer(args['n_envs'], self.seq_len, self.n_minibatches, self.minibach_size,
                                self.env_vec.state_space, self.env_vec.action_dim)

        if self.use_lstm:
            self.last_lstm_state = None

        with self.graph.as_default():
            self.variables = ray.experimental.TensorFlowVariables(self.ac.loss, self.sess)

    """ Implementation """
    def _build_graph(self):
        self.env_phs = self._setup_env_placeholders(self.env_vec.state_space, self.env_vec.action_dim)

        self.ac = ActorCritic('ac', 
                              self.args['ac'], 
                              self.graph,
                              self.env_vec, 
                              self.env_phs,
                              self.minibach_size,
                              self.name, 
                              log_tensorboard=self.log_tensorboard,
                              log_params=self.log_params)

    def _setup_env_placeholders(self, state_space, action_dim):
        env_phs = {}

        with tf.name_scope('placeholder'):
            env_phs['state'] = tf.placeholder(tf.float32, shape=[None, *state_space], name='state')
            env_phs['return'] = tf.placeholder(tf.float32, shape=[None, 1], name='return')
            env_phs['value'] = tf.placeholder(tf.float32, shape=[None, 1], name='value')
            env_phs['advantage'] = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
            env_phs['old_logpi'] = tf.placeholder(tf.float32, shape=[None, 1], name='old_logpi')
            env_phs['entropy_coef'] = tf.placeholder(tf.float32, shape=None, name='entropy_coeff')
        
        return env_phs

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('ppo_loss_', self.ac.ppo_loss)
                tf.summary.scalar('entropy_', self.ac.entropy)
                tf.summary.scalar('V_loss_', self.ac.V_loss)
                
            
            with tf.name_scope('V'):
                tf.summary.scalar('max_Q_with_actor', tf.reduce_max(self.ac.V))
                tf.summary.scalar('min_Q_with_actor', tf.reduce_min(self.ac.V))
                tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.ac.V))

    """ code for single agent """
    def optimize(self):
        fetches = [self.ac.opt_op, 
                   [self.ac.ppo_loss, self.ac.entropy, 
                    self.ac.V_loss, self.ac.loss, 
                    self.ac.approx_kl, self.ac.clipfrac]]
        feed_dict = {}
        if self.use_lstm:
            fetches.append(self.ac.final_state)
            feed_dict.update({k: v for k, v in zip(self.ac.initial_state, self.last_lstm_state)})

        # normalize advantages (& returns)
        feed_dict.update({
            self.env_phs['state']: self.buffer.get_flat_batch('state', self.minibatch_idx),
            self.ac.action: self.buffer.get_flat_batch('action', self.minibatch_idx),
            self.env_phs['return']: self.buffer.get_flat_batch('return', self.minibatch_idx),
            self.env_phs['value']: self.buffer.get_flat_batch('value', self.minibatch_idx),
            self.env_phs['advantage']: self.buffer.get_flat_batch('advantage', self.minibatch_idx),
            self.env_phs['old_logpi']: self.buffer.get_flat_batch('old_logpi', self.minibatch_idx),
            self.env_phs['entropy_coef']: self.entropy_coef
        })

        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_lstm:
            _, loss_info, self.last_lstm_state = results
        else:
            _, loss_info = results
        self.minibatch_idx = self.minibatch_idx + 1 if self.minibatch_idx + 1 < self.n_minibatches else 0
        
        return loss_info

    def sample_trajectories(self):
        env_stats = self._sample_data()
        self.buffer.compute_ret_adv(self.args['advantage_type'], self.gamma, self.gae_discount)
        if self.args['advantage_type'] == 'gae':
            self.buffer.normalize_adv(np.mean(self.buffer['advantage']), np.std(self.buffer['advantage']))
        
        return env_stats

    def act(self, state):
        fetches = [self.ac.action, self.ac.V, self.ac.logpi]
        feed_dict = {self.env_phs['state']: state}
        if self.ac.use_lstm:
            fetches.append(self.ac.final_state)
            feed_dict.update({k: v for k, v in zip(self.ac.initial_state, self.last_lstm_state)})
        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_lstm:
            action, value, logpi, self.last_lstm_state = results
        else:
            action, value, logpi = results

        return action, np.squeeze(value), np.squeeze(logpi)

    """ Implementation """
    def _sample_data(self):
        self.buffer.reset()
        state = self.env_vec.reset()

        if self.use_lstm:
            # set initial state to zeros for every epochs
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: state})
        
        for _ in range(self.seq_len):
            action, value, logpi = self.act(state)
            next_state, reward, done, _ = self.env_vec.step(action)

            self.buffer.add(state, action, reward, value, logpi, 1-np.array(done))

            state = next_state
        
        # add one more ad hoc value so that we can take values[1:] as next state values
        value = np.squeeze(self.sess.run(self.ac.V, feed_dict={self.env_phs['state']: state}))
        self.buffer['value'][:, -1] = value
        
        return self.env_vec.get_episode_score(), self.env_vec.get_episode_length()
