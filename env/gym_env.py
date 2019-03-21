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

class GymEnvironment():
    def __init__(self, name, model_name=None, atari=False, seed=0):
        self.env = gym.make(name)
        if model_name is None:
            model_name = name
        # self.env = gym.wrappers.Monitor(self.env, f'data/gym/{model_name}', force=True)
        if atari:
            self.env = atari_wrappers.wrap_deepmind(self.env)
        self.env.seed(seed)

        self.state_space = self.env.observation_space.shape
        self.is_action_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_space = self.env.action_space.n if self.is_action_discrete else self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dist_type = action_dist_type(self.env)
        
        self.max_episode_steps = self.env.spec.max_episode_steps

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def is_inrange(self, action):
        return action > self.action_low and action < self.action_high
