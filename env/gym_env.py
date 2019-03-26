import numpy as np
import gym
import ray

from utility import tf_distributions
from env import atari_wrappers


def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError

class GymEnvironment:
    def __init__(self, name, max_episode_steps=None, atari=False, seed=0):
        self.env = gym.make(name)
        # if model_name:
        #     self.env = gym.wrappers.Monitor(self.env, f'data/gym/{model_name}', force=True)
        if atari:
            self.env = atari_wrappers.wrap_deepmind(self.env)
        self.env.seed(seed)

        self.state_space = self.env.observation_space.shape
        self.is_action_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_space = self.env.action_space.n if self.is_action_discrete else self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dist_type = action_dist_type(self.env)
        
        self.max_episode_steps = self.env.spec.max_episode_steps if max_episode_steps is None else max_episode_steps

        self.score = 0
        self.early_done = False

    def reset(self):
        print('reset')
        self.score = 0
        self.early_done = False

        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.score += 0 if self.early_done else reward
        self.early_done = done
        return (next_state, reward, done, info)

    def render(self):
        return self.env.render()

    def is_inrange(self, action):
        return action > self.action_low and action < self.action_high

    def get_score(self):
        return self.score

RayGymEnv = ray.remote(GymEnvironment)

class GymEnvVec:
    def __init__(self, name, n_envs=1, max_episode_steps=None, atari=False, seed=0):
        self.envs = [RayGymEnv.remote(name, max_episode_steps, atari, seed + 10 * i) for i in range(n_envs)]

        env = gym.make(name)
        self.state_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = n_envs
        self.max_episode_steps = env.spec.max_episode_steps if max_episode_steps is None else max_episode_steps

    def reset(self):
        return ray.get([env.reset.remote() for env in self.envs])
    
    def step(self, actions):
        step_info = list(zip(*ray.get([env.step.remote(a) for env, a in zip(self.envs, actions)])))
        return step_info

    def get_score(self):
        return ray.get([env.get_score.remote() for env in self.envs])
