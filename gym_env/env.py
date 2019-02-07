import gym
import ray

from utility import tf_distributions

def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError

class GymEnvironment():
    def __init__(self, name, seed=0):
        self.env = gym.make(name)
        self.env.seed(seed)

        self.observation_dim = self.env.observation_space.shape[0]
        self.is_action_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_dim = self.env.action_space.n if self.is_action_discrete else self.env.action_space.shape[0]

        self.action_dist_type = action_dist_type(self.env)
        self.max_episode_steps = self.env.spec.max_episode_steps

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()