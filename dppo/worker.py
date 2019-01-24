import numpy as np
import ray

from utils.np_math import norm
from agent import Agent

@ray.remote
class Worker(Agent):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 reuse=False,
                 save=True):
        super().__init__(name, args, env_args, sess_config=sess_config, 
                         reuse=reuse, save=save)

        self.obs, self.actions, self.target_V, self.advantages = [], [], [], []

    @ray.method(num_return_vals=2)
    def compute_loss(self):
        self.indices = np.random.choice(len(self.obs), self._batch_size)

        sample_obs = self.obs[self.indices]
        sample_actions = self.actions[self.indices]
        sample_returns = self.target_V[self.indices]
        sample_advantages = self.advantages[self.indices]

        _, actor_loss, critic_loss = self.sess.run(
            [self.opt_op, self.actor.loss, self.critic.loss],
                feed_dict={
                self.env_phs['observation']: sample_obs,
                self.env_phs['action']: sample_actions,
                self.env_phs['target_V']: sample_returns,
                self.env_phs['advantage']: sample_advantages
            })

        return actor_loss, critic_loss

    def compute_gradients(self, weights):
        self.set_weights(weights)

        self.indices = np.random.choice(len(self.obs), self._batch_size)
        sample_obs = self.obs[self.indices]
        sample_actions = self.actions[self.indices]
        sample_returns = self.target_V[self.indices]
        sample_advantages = self.advantages[self.indices]

        grads = self.sess.run(
            [grad_and_var[0] for grad_and_var in self.grads_and_vars],
            feed_dict={
                self.env_phs['observation']: sample_obs,
                self.env_phs['action']: sample_actions,
                self.env_phs['target_V']: sample_returns,
                self.env_phs['advantage']: sample_advantages
            })
        
        return grads

    def sample_trajectories(self, weights):
        self.set_weights(weights)

        self.clear_data()
        n_episodes = 0
        rewards, nonterminal = [], []
        while len(self.obs) < self._n_updates_per_iteration * self._batch_size:
            ob = self.env.reset()
            for _ in range(self._max_path_length):
                self.obs.append(ob)
                action = self.act(ob)
                self.actions.append(action)
                ob, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                nonterminal.append(1 - done)
                
                if done:
                    break
            n_episodes += 1

        self.obs.append(ob) # add one more fake observation so that we can take obs[1:] as next observations

        rewards = np.array(rewards)
        nonterminal = np.array(nonterminal)

        V = self.sess.run(self.critic.V, feed_dict={self.env_phs['observation']: self.obs})

        self.target_V = np.copy(rewards)
        self.target_V[-1] = rewards[-1]

        # shaped_rewards = rewards + nonterminal * self._gamma * V[1:] - V[:-1]
        # self.advantages = np.copy(rewards)
        # self.advantages[-1] = shaped_rewards[-1]

        for i in reversed(range(len(rewards)-1)):
            self.target_V[i] += nonterminal[i] * self._gamma * self.target_V[i+1]
            # self.advantages[i] += nonterminal[i] * self._advantage_discount * self.advantages[i+1]
        
        # code for test
        V = norm(V, np.mean(self.target_V), np.std(self.target_V))
        self.advantages = norm(self.target_V - V[:-1])
        self.target_V = norm(self.target_V)
        # end

        self.obs.pop()  # remove the ad hoc observation
        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)

        score = np.sum(rewards) / n_episodes

        return score

    def set_weights(self, weights):
        self.variables.set_flat(weights)
        
    def clear_data(self):
        self.obs, self.actions, self.target_V, self.advantages = [], [], [], []

    def act(self, observation):
        observation = observation.reshape((-1, self.env.observation_dim))
        action = self.sess.run(self.action, feed_dict={self.env_phs['observation']: observation})

        return action