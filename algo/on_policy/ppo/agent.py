import os
import numpy as np
import tensorflow as tf
from ray.experimental.tf_utils import TensorFlowVariables

from env.gym_env import GymEnv, GymEnvVec
from utility.losses import huber_loss
from basic_model.model import Model
from algo.on_policy.ppo.networks import ActorCritic
from algo.on_policy.ppo.buffer import PPOBuffer
from utility.tf_utils import stats_summary


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
                 device=None,
                 reuse=None,
                 graph=None):
        # hyperparameters
        self.gamma = args['gamma']
        self.gae_discount = self.gamma * args['lam']
        self.seq_len = args['seq_len']
        self.use_rnn = args['ac']['use_rnn']

        self.entropy_coef = args['ac']['entropy_coef']
        self.n_minibatches = args['n_minibatches']
        self.minibach_size = self.seq_len // self.n_minibatches
        self.minibatch_idx = 0

        # environment info
        self.env_vec = GymEnvVec(env_args) if env_args['n_envs'] > 1 else GymEnv(env_args)
        super().__init__(name, args, 
                         sess_config=sess_config,
                         save=save, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_score=log_score,
                         device=device,
                         reuse=reuse,
                         graph=graph)

        self.buffer = PPOBuffer(env_args['n_envs'], self.seq_len, self.n_minibatches, self.minibach_size,
                                self.env_vec.state_space, self.env_vec.action_dim, args['mask'])

        if self.use_rnn:
            self.last_lstm_state = None

        with self.graph.as_default():
            self.variables = TensorFlowVariables(self.ac.loss, self.sess)

    """ code for single agent """
    def optimize(self):
        get_data = lambda name: self.buffer.get_flat_batch(name, self.minibatch_idx)

        if self.minibatch_idx == 0 and self.use_rnn:
            # set initial state to zeros for every pass of buffer
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: get_data('state')})
            
        # construct fetches and feed_dict
        fetches = [self.ac.opt_op, 
                   [self.ac.ppo_loss, self.ac.entropy, 
                    self.ac.V_loss, self.ac.loss, 
                    self.ac.approx_kl, self.ac.clipfrac]]
        feed_dict = {
            self.env_phs['state']: get_data('state'),
            self.ac.action: get_data('action'),
            self.env_phs['return']: get_data('return'),
            self.env_phs['value']: get_data('value'),
            self.env_phs['advantage']: get_data('advantage'),
            self.env_phs['old_logpi']: get_data('old_logpi'),
            self.env_phs['entropy_coef']: self.entropy_coef
        }
        if self.use_rnn:
            fetches.append(self.ac.final_state)
            feed_dict.update({k: v for k, v in zip(self.ac.initial_state, self.last_lstm_state)})
        fetches.append([self.ac.opt_step, self.graph_summary])

        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_rnn:   # assuming log_tensorboard=True for simplicity, since optimize is only called by learner
            _, loss_info, self.last_lstm_state, (opt_step, summary) = results
        else:
            _, loss_info, (opt_step, summary) = results

        self.minibatch_idx = self.minibatch_idx + 1 if self.minibatch_idx + 1 < self.n_minibatches else 0

        if opt_step % (self.args['n_updates'] * self.args['n_minibatches']) == 0:
            self.writer.add_summary(summary, opt_step)
            self.save()

        return loss_info

    def sample_trajectories(self):
        env_stats = self._sample_data()
        self.buffer.compute_ret_adv(self.args['advantage_type'], self.gamma, self.gae_discount)
        if self.args['advantage_type'] == 'gae':
            self.buffer.normalize_adv(self.buffer['advantage'].mean(), self.buffer['advantage'].std())
        
        return env_stats

    def act(self, state):
        state = np.reshape(state, (-1, *self.env_vec.state_space))
        fetches = [self.ac.action, self.ac.V, self.ac.logpi]
        feed_dict = {self.env_phs['state']: state}
        if self.use_rnn:
            fetches.append(self.ac.final_state)
            feed_dict.update({k: v for k, v in zip(self.ac.initial_state, self.last_lstm_state)})
        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_rnn:
            action, value, logpi, self.last_lstm_state = results
        else:
            action, value, logpi = results

        return action, np.squeeze(value), np.squeeze(logpi)

    def demonstrate(self):
        state = self.env_vec.reset()
        if self.use_rnn:
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: state})
        
        for _ in range(self.seq_len):
            self.env_vec.render()
            action, _, _ = self.act(state)
            state, _, done, _ = self.env_vec.step(action)

            if done:
                break

        print(f'Demonstration score:\t{self.env_vec.get_episode_score()}')
        print(f'Demonstration length:\t{self.env_vec.get_episode_length()}')

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
        
        self._log_loss()

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

            with tf.name_scope('value'):
                stats_summary(self.ac.V, 'V')
                stats_summary(self.env_phs['advantage'], 'advantage')
                stats_summary(self.env_phs['return'], 'return')

            with tf.name_scope('policy'):
                stats_summary(self.ac.action_distribution.mean, 'mean')
                stats_summary(self.ac.action_distribution.std, 'std')
                stats_summary(self.ac.action_distribution.entropy(), 'entropy')

    def _sample_data(self):
        self.buffer.reset()
        state = self.env_vec.reset()
        
        if self.use_rnn:
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
