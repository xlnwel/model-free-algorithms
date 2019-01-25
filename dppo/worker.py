import numpy as np
import ray

from utils.np_math import norm
from agent import Agent

@ray.remote
class Worker(Agent):
    """ Interface """
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=True):
        super().__init__(name, args, env_args, sess_config=sess_config, 
                         reuse=reuse, save=save)

        self._init_data()

    def compute_gradients(self, weights):
        assert (isinstance(self.obs, np.ndarray)
                and isinstance(self.returns, np.ndarray)
                and isinstance(self.advantages, np.ndarray) 
                and isinstance(self.old_neglogpi, np.ndarray))

        self._set_weights(weights)

        indices = np.random.choice(len(self.obs), self._batch_size)
        sample_obs = self.obs[indices]
        sample_actions = self.actions[indices]
        sample_returns = self.returns[indices]
        sample_advantages = self.advantages[indices]
        sample_old_neglogpi = self.old_neglogpi[indices]

        grads = self.sess.run(
            [grad_and_var[0] for grad_and_var in self.grads_and_vars],
            feed_dict={
                self.env_phs['observation']: sample_obs,
                self.actor.action: sample_actions,
                self.env_phs['return']: sample_returns,
                self.env_phs['advantage']: sample_advantages,
                self.actor.old_neglogpi_ph: sample_old_neglogpi
            })
        
        return grads

    def sample_trajectories(self, weights):
        self._set_weights(weights)
        self._init_data()

        n_episodes = 0
        values, rewards, nonterminals = [], [], []
        while len(self.obs) < self._n_updates_per_iteration * self._batch_size:
            ob = self.env.reset()

            for _ in range(self._max_path_length):
                self.obs.append(ob)
                action, value, neglogpi = self.step(ob)
                ob, reward, done, _ = self.env.step(action)

                self.actions.append(action)
                values.append(value)
                self.old_neglogpi.append(neglogpi)
                rewards.append(reward)
                nonterminals.append(1 - done)

                if done:
                    break
            n_episodes += 1

        score = np.sum(rewards) / n_episodes
        
        # add one more ad hoc state value so that we can take values[1:] as next state values
        if done:
            ob = self.env.reset()
        _, value, _ = self.step(ob)
        values.append(value)

        rewards = np.asarray(rewards, dtype=np.float32)
        nonterminals = np.asarray(nonterminals, dtype=np.uint8)

        # shaped_rewards = rewards + nonterminals * self._gamma * values[1:] - values[:-1]
        # self.advantages = shaped_rewards

        self.returns = rewards
        self.returns[-1] += nonterminals[-1] * self._gamma * values[-1]

        for i in reversed(range(len(rewards)-1)):
            self.returns[i] += nonterminals[i] * self._gamma * self.returns[i+1]
            # self.advantages[i] += nonterminals[i] * self._advantage_discount * self.advantages[i+1]
        
        # normalized advantages
        values = norm(values[:-1], np.mean(self.returns), np.std(self.returns))
        self.advantages = norm(self.returns - values)
        self.returns = norm(self.returns)
        # end

        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.actions = np.asarray(self.actions)
        self.old_neglogpi = np.asarray(self.old_neglogpi, dtype=np.float32)

        return score

    def step(self, observation):
        observation = np.reshape(observation, (-1, self.env.observation_dim))
        action, value, neglogpi = self.sess.run([self.action, self.critic.V, self.actor.neglogpi], 
                                            feed_dict={self.env_phs['observation']: observation})

        return action, value, neglogpi

    """ Implementation """
    def _set_weights(self, weights):
        self.variables.set_flat(weights)
        
    def _init_data(self):
        self.obs, self.actions, self.returns, self.advantages, self.old_neglogpi = [], [], [], [], []