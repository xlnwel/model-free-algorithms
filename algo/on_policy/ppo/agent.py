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
from utility.utils import pwc


class Agent(Model):
    """ Interface """
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 save=False,
                 log=False,
                 log_tensorboard=False,
                 log_params=False,
                 log_stats=False,
                 device=None,
                 reuse=None,
                 graph=None):
        # hyperparameters
        self.gamma = args['gamma']
        self.gae_discount = self.gamma * args['lam']
        self.seq_len = args['seq_len']
        self.n_minibatches = args['n_minibatches']
        self.mask_data = args['mask_data']
        self.mask_loss = args['mask_loss']

        self.use_lstm = args['ac']['use_lstm']
        self.entropy_coef = args['ac']['entropy_coef']
        self.n_value_updates = args['ac']['n_value_updates']
        self.minibatch_idx = 0

        # environment info
        self.env_vec = (GymEnvVec(env_args) if env_args['n_envs'] > 1 
                        else GymEnv(env_args))

        self.buffer = PPOBuffer(env_args['n_envs'], self.seq_len, self.n_minibatches,
                                self.env_vec.state_space, self.env_vec.action_dim, 
                                self.mask_data or self.mask_loss)

        super().__init__(name, args, 
                         sess_config=sess_config,
                         save=save, 
                         log=log,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_stats=log_stats,
                         device=device,
                         reuse=reuse,
                         graph=graph)

        if self.use_lstm:
            self.last_lstm_state = None

        with self.graph.as_default():
            self.variables = TensorFlowVariables([self.ac.policy_loss, self.ac.V_loss], self.sess)

    def sample_trajectories(self):
        env_stats = self._sample_data()

        return env_stats

    def act(self, state):
        state = np.reshape(state, (-1, *self.env_vec.state_space))
        fetches = [self.ac.action, self.ac.V, self.ac.logpi]
        feed_dict = {self.env_phs['state']: state}
        if self.use_lstm:
            fetches.append(self.ac.final_state)
            feed_dict.update({k: v for k, v in zip(self.ac.initial_state, self.last_lstm_state)})
        results = self.sess.run(fetches, feed_dict=feed_dict)
        if self.use_lstm:
            action, value, logpi, self.last_lstm_state = results
        else:
            action, value, logpi = results

        return action, np.squeeze(value), np.squeeze(logpi)

    def demonstrate(self):
        state = self.env_vec.reset()
        state = np.reshape(state, (-1, *self.env_vec.state_space))
        if self.use_lstm:
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: state})
        
        for _ in range(self.env_vec.max_episode_steps):
            action, _, _ = self.act(state)
            state, _, done, _ = self.env_vec.step(action)

            if done:
                break

        pwc(f'Demonstration score:\t{self.env_vec.get_score()}', 'green')
        pwc(f'Demonstration length:\t{self.env_vec.get_length()}', 'green')

    def shuffle_buffer(self):
        if not self.use_lstm:
            self.buffer.shuffle()

    def optimize(self, epoch_i):
        loss_info_list = []
        for i in range(self.args['n_updates']):
            self.shuffle_buffer()
            for j in range(self.args['n_minibatches']):
                loss_info, summary = self._optimize(epoch_i)

                loss_info_list.append(loss_info)

                if 'max_kl' not in self.args or self.args['max_kl'] == 0.:
                    continue
                kl = np.mean(loss_info[2])
                if kl > self.args['max_kl']:
                    pwc(f'{self.model_name}: Eearly stopping at epoch-{epoch_i} update-{i} minibatch-{j} due to reaching max kl.\nCurrent kl={kl}')
                    break
            if 'max_kl' not in self.args or self.args['max_kl'] == 0.:
                continue
            
            if kl > self.args['max_kl']:
                break

        if self.log_tensorboard:
            self.writer.add_summary(summary, epoch_i)
        if hasattr(self, 'saver'):
            self.save()

        return loss_info_list

    def print_construction_complete(self):
        pwc(f'{self.name} has been constructed.', color='cyan')

    """ Implementation """
    def _build_graph(self):
        self.env_phs = self._setup_env_placeholders(self.env_vec.state_space, self.env_vec.action_dim)

        self.args['ac']['batch_seq_len'] = self.seq_len // self.n_minibatches
        self.ac = ActorCritic('ac', 
                              self.args['ac'], 
                              self.graph,
                              self.env_vec, 
                              self.env_phs,
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
            env_phs['entropy_coef'] = tf.placeholder(tf.float32, shape=None, name='entropy_coef')
            if self.mask_loss:
                env_phs['mask_loss'] = tf.placeholder(tf.float32, shape=[None, 1], name='mask_loss')
            else:
                env_phs['mask_loss'] = None
        
        return env_phs

    def _sample_data(self):
        self.buffer.reset()
        state = self.env_vec.reset()
        
        if self.use_lstm:
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: state})
        
        for _ in range(self.seq_len):
            action, value, logpi = self.act(state)
            next_state, reward, done, _ = self.env_vec.step(action)
            
            data = dict(state=state, action=action, reward=reward, value=value, old_logpi=logpi, nonterminal=1-np.asarray(done))
            if self.mask_data or self.mask_loss:
                mask = self.env_vec.get_mask()
                data['mask'] = mask

            self.buffer.add(data)

            state = next_state
        
        # add one more ad hoc value so that we can take values[1:] as next state values
        last_value = np.squeeze(self.sess.run(self.ac.V, feed_dict={self.env_phs['state']: state}))

        self.buffer.compute_ret_adv(last_value, self.args['advantage_type'], self.gamma, self.gae_discount, self.mask_data)
        
        return self.env_vec.get_score(), self.env_vec.get_length()

    def _optimize(self, timestep=None):
        # construct policy fetches
        policy_fetches = [self.ac.policy_optop, 
                          [self.ac.ppo_loss, self.ac.entropy, 
                           self.ac.approx_kl, self.ac.clipfrac]]
        if self.use_lstm:
            policy_fetches.append(self.ac.final_state)
        if self.log_tensorboard:
            policy_fetches.append(self.graph_summary)

        # construct feed_dict
        feed_dict = self._get_feeddict(timestep)

        results = self.sess.run(policy_fetches, feed_dict=feed_dict)
        summary = None    # default values if self.log_tensorboard is None
        if self.use_lstm and self.log_tensorboard:   # assuming log_tensorboard=True for simplicity, since optimize() is only called by learner
            _, loss_info, self.last_lstm_state, summary = results
        elif self.use_lstm:
            _, loss_info, self.last_lstm_state = results
        elif self.log_tensorboard:
            _, loss_info, summary = results
        else:
            _, loss_info = results

        v_fetches = [self.ac.v_optop, self.ac.V_loss]
        for _ in range(self.n_value_updates):
            _, v_loss = self.sess.run(v_fetches, feed_dict=feed_dict)
        
        loss_info.append(v_loss)
        self.minibatch_idx = (self.minibatch_idx + 1) % self.n_minibatches

        return loss_info, summary
        

    def _get_feeddict(self, timestep=None):
        get_data = lambda name: self.buffer.get_flat_batch(name, self.minibatch_idx)

        if self.minibatch_idx == 0 and self.use_lstm:
            # set initial state to zeros for every pass of buffer
            self.last_lstm_state = self.sess.run(self.ac.initial_state, feed_dict={self.env_phs['state']: get_data('state')})
            
        # construct feed_dict
        feed_dict = {
            self.env_phs['state']: get_data('state'),
            self.ac.action: get_data('action'),
            self.env_phs['return']: get_data('return'),
            self.env_phs['value']: get_data('value'),
            self.env_phs['advantage']: get_data('advantage'),
            self.env_phs['old_logpi']: get_data('old_logpi'),
            self.env_phs['entropy_coef']: self.entropy_coef
        }
        
        if self.use_lstm:
            feed_dict.update({k: v for k, v in zip(self.ac.initial_state, self.last_lstm_state)})
        if self.mask_loss:
            feed_dict[self.env_phs['mask_loss']] = get_data('mask')

        return feed_dict

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('ppo_loss_', self.ac.ppo_loss)
                tf.summary.scalar('entropy_', self.ac.entropy)
                tf.summary.scalar('V_loss_', self.ac.V_loss)

            with tf.name_scope('value'):
                stats_summary('V', self.ac.V)
                stats_summary('advantage', self.env_phs['advantage'])
                stats_summary('return', self.env_phs['return'])
                
                if self.env_phs['mask_loss'] is not None:
                    stats_summary('V_mask', self.ac.V * self.env_phs['mask_loss'])
                    stats_summary('advantage_mask', self.env_phs['advantage']* self.env_phs['mask_loss'])
                    stats_summary('return_mask', self.env_phs['return']* self.env_phs['mask_loss'])

            if self.mask_loss:
                with tf.name_scope('mask'):
                    stats_summary('mask', self.env_phs['mask_loss'])

            with tf.name_scope('policy'):
                stats_summary('mean_', self.ac.action_distribution.mean)
                stats_summary('std_', self.ac.action_distribution.std)
                stats_summary('entropy_', self.ac.action_distribution.entropy())