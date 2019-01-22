import numpy as np
import ray

from utils.utils import norm
from ppo_gae import PPOGAE

@ray.remote
class Worker(PPOGAE):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=True):
        super().__init__(name, args, env_args, sess_config=sess_config, 
                         reuse=reuse, save=save)

        self.obs, self.actions, self.returns, self.advantages = [], [], [], []

    def compute_gradients(self, weights):
        self.set_weights(weights)

        indices = np.random.choice(len(self.obs), self._batch_size)
        sample_obs = self.obs[indices]
        sample_actions = self.actions[indices]
        sample_returns = self.returns[indices]
        sample_advantages = self.advantages[indices]

        grads = self.sess.run(
            [grad_and_var[0] for grad_and_var in self.grads_and_vars],
            feed_dict={
                self.env_phs['observations']: sample_obs,
                self.env_phs['actions']: sample_actions,
                self.env_phs['returns']: sample_returns,
                self.env_phs['advantages']: sample_advantages
            })
        
        return grads

    def sample_trajectories(self, weights=None):
        self.set_weights(weights)

        self.clear_data()
        score = 0
        n_episodes = 0
        while len(self.obs) < self._n_updates_per_iteration * self._batch_size:
            obs, next_obs, rewards, dones = [], [], [], []
            ob = self.env.reset()
            for _ in range(self._max_path_length):
                obs.append(ob)
                action = self.act(ob)
                self.actions.append(action)
                ob, reward, done, _ = self.env.step(action)
                next_obs.append(ob)
                rewards.append(reward)
                dones.append(done)
                score += reward
                if done:
                    break
            n_episodes += 1

            Vs = self.sess.run(self.critic.V, feed_dict={self.env_phs['observations']: obs})
            next_Vs = self.sess.run(self.critic.V, feed_dict={self.env_phs['observations']: next_obs})
            Vs, next_Vs = np.squeeze(Vs), np.squeeze(next_Vs)  
            deltas = np.array(rewards) + (1 - np.array(dones)) * self._gamma * next_Vs - Vs

            self.obs += obs
            self.returns.insert(0, rewards[-1])
            self.advantages.insert(0, deltas[-1])

            for r, d in zip(reversed(rewards[:-1]), reversed(deltas[:-1])):
                self.returns.insert(0, r + self._gamma * self.returns[0])
                self.advantages.insert(0, d + self._advantage_discount * self.advantages[0])
        
        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)
        self.returns = np.array(self.returns)
        self.advantages = np.array(self.advantages)
        score /= n_episodes

        return score

    def set_weights(self, weights):
        self.variables.set_flat(weights)
        
    def clear_data(self):
        self.obs, self.actions, self.returns, self.advantages = [], [], [], []

    def act(self, observations):
        observations = observations.reshape((-1, self.observation_dim))
        action = self.sess.run(self.action, feed_dict={self.actor.observation_ph: observations})

        return action