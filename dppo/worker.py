import numpy as np
import ray

from utils.np_math import norm
from agent import Agent


@ray.remote#(num_cpus=0.5, num_gpus=0.04)
class Worker(Agent):
    """ Interface """
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=False,
                 device=None):
        super().__init__(name, args, env_args, sess_config=sess_config, 
                         reuse=reuse, save=save, device=device)

        self._shuffle = self._args['option']['shuffle']
        self._init_data()

    @ray.method(num_return_vals=2)
    def compute_gradients(self, weights):
        assert (isinstance(self.obs, np.ndarray)
                and isinstance(self.actions, np.ndarray)
                and isinstance(self.returns, np.ndarray)
                and isinstance(self.advantages, np.ndarray) 
                and isinstance(self.old_neglogpi, np.ndarray))

        self._set_weights(weights)

        start_idx, end_idx = self.batch_i * self._minibatch_size, (self.batch_i + 1) * self._minibatch_size
        sample_obs = self.obs[start_idx: end_idx]
        sample_actions = self.actions[start_idx: end_idx]
        sample_returns = self.returns[start_idx: end_idx]
        sample_advantages = self.advantages[start_idx: end_idx]
        sample_old_neglogpi = self.old_neglogpi[start_idx: end_idx]
        self.batch_i += 1

        grads, loss_info = self.sess.run(
            [[grad_and_var[0] for grad_and_var in self.grads_and_vars], 
            [self.actor.ppo_loss, self.actor.entropy_loss, self.actor.loss, self.critic.loss, self.loss]],
            feed_dict={
                self.env_phs['observation']: sample_obs,
                self.action: sample_actions,
                self.env_phs['return']: sample_returns,
                self.env_phs['advantage']: sample_advantages,
                self.actor.old_neglogpi_ph: sample_old_neglogpi
            })
        
        return grads, loss_info

    def sample_trajectories(self, weights):
        # helper functions
        def sample_data(env, batch_size, max_path_length):
            obs, actions, values, rewards, old_neglogpi, nonterminals = [], [], [], [], [], []

            n_episodes = 0
            scores = []
            while len(obs) < batch_size:
                ob = env.reset()
                score = 0
                for _ in range(max_path_length):
                    obs.append(ob)
                    action, value, neglogpi = self.step(ob)
                    ob, reward, done, _ = env.step(action)
                    score += reward

                    actions.append(action)
                    values.append(value)
                    old_neglogpi.append(neglogpi)
                    rewards.append(reward)
                    nonterminals.append(1 - done)

                    if done:
                        break
                n_episodes += 1
                scores.append(score)
                nonterminals[-1] = 0
            
            # add one more ad hoc state value so that we can take values[1:] as next state values
            if done:
                ob = self.env.reset()
            _, value, _ = self.step(ob)
            values.append(value)

            return scores, (np.asarray(obs, dtype=np.float32),
                            np.reshape(actions, [len(obs), env.action_dim]),
                            np.squeeze(old_neglogpi),
                            np.asarray(rewards, dtype=np.float32),
                            np.asarray(values, dtype=np.float32),
                            np.asarray(nonterminals, dtype=np.uint8))
        
        def compute_returns_advantages(rewards, values, nonterminals, gamma):
            adv_type =self._args['option']['advantage_type']
            if adv_type == 'norm':
                returns = rewards
                next_return = 0
                for i in reversed(range(len(rewards))):
                    returns[i] = rewards[i] + nonterminals[i] * gamma * next_return
                    next_return = returns[i]

                # normalize returns and advantages
                values = norm(values[:-1], np.mean(returns), np.std(returns))
                advantages = norm(returns - values)
                returns = norm(returns)
            elif adv_type == 'gae':
                advantages = np.zeros_like(rewards)
                last_adv = 0
                for i in reversed(range(len(rewards))):
                    delta = rewards[i] + nonterminals[i] * self._gamma * values[i+1] - values[i]
                    advantages[i] = last_adv = delta + nonterminals[i] * self._advantage_discount * last_adv
                # deltas = rewards + nonterminals * self._gamma * values[1:] - values[:-1]
                # advantages = deltas
                # for i in reversed(range(len(rewards) - 1)):
                #     advantages[i] += nonterminals[i] * self._advantage_discount * advantages[i+1]
                returns = advantages + values[:-1]

                # normalize returns and advantages
                # values = norm(values[:-1], np.mean(returns), np.std(returns))
                advantages = norm(advantages)
                # returns = norm(returns)
            else:
                NotImplementedError
            
            return returns, advantages

        # function content
        self._set_weights(weights)
        self._init_data()

        batch_size = self._n_minibatches * self._minibatch_size
        score_info, data = sample_data(self.env,
                                    batch_size,
                                    self._max_path_length)
        obs, actions, old_neglogpi, rewards, values, nonterminals = data

        returns, advantages = compute_returns_advantages(rewards, values, nonterminals, self._gamma)

        if self._shuffle:
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            self.obs = obs[indices]
            self.actions = actions[indices]
            self.old_neglogpi = old_neglogpi[indices]
            self.returns = returns[indices]
            self.advantages = advantages[indices]
            # print('type(self.obs)', type(self.obs))
            # print('type(self.actions)', type(self.actions))
            # print('type(self.old_neglogpi)', type(self.old_neglogpi))
            # print('type(self.returns)', type(self.returns))
            # print('type(self.advantages)', type(self.advantages))
        else:
            self.obs = obs
            self.actions = actions
            self.old_neglogpi = old_neglogpi
            self.returns = returns
            self.advantages = advantages
        self.batch_i = 0

        return score_info

    def step(self, observation):
        observation = np.reshape(observation, (-1, self.env.observation_dim))
        action, value, neglogpi = self.sess.run([self.action, self.critic.V, self.actor.neglogpi], 
                                            feed_dict={self.env_phs['observation']: observation})
        return np.reshape(action, [-1]), value, neglogpi

    """ Implementation """
    def _set_weights(self, weights):
        self.variables.set_flat(weights)

    def _init_data(self):
        self.obs, self.actions, self.returns, self.advantages, self.old_neglogpi = [], [], [], [], []
