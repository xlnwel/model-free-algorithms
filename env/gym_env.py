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

class envstats:
    def __init__(self, env):
        self.env = env
        self.score = 0
        self.eps_len = 0
        self.early_done = 0
        
        self.env_reset = env.reset
        self.env.reset = self.reset
        self.env_step = env.step
        self.env.step = self.step

        self.env.get_episodic_score = lambda _: self.score
        self.env.get_episodic_length = lambda _: self.eps_len

    def __call__(self, *args, **kwargs):
        self.env = self.env(*args, **kwargs)

        return self.env

    def reset(self):
        self.score = 0
        self.eps_len = 0
        self.early_done = 0
        
        return self.env_reset(self.env)

    def step(self, action):
        next_state, reward, done, info = self.env_step(self.env, action)
        self.score += np.where(self.early_done, 0, reward)
        self.eps_len += np.where(self.early_done, 0, 1)
        self.early_done = np.array(done)

        return next_state, reward, done, info

@envstats
class GymEnv:
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

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
        
    def render(self):
        return self.env.render()

    def is_inrange(self, action):
        return action > self.action_low and action < self.action_high

@envstats
class GymEnvVec:
    def __init__(self, name, n_envs=1, max_episode_steps=None, atari=False, seed=0):
        self.envs = [gym.make(name) for i in range(n_envs)]
        [env.seed(seed + 10 * i) for i, env in enumerate(self.envs)]

        env = self.envs[0]
        self.state_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = n_envs
        self.max_episode_steps = env.spec.max_episode_steps if max_episode_steps is None else max_episode_steps

        self.score = np.zeros(n_envs)
        self.eps_len = np.zeros(n_envs)
        self.early_done = np.zeros(n_envs)
        self.zeros = np.zeros(n_envs)

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        return list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))

