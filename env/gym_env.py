import numpy as np
import gym
import ray

from utility import tf_distributions
from env import atari_wrappers
from utility.utils import colorize


def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError

class envstats:
    """ Provide Environment Stats Records """
    def __init__(self, env):
        self.EnvType = env
        self.score = 0
        self.eps_len = 0
        self.early_done = 0
        
        self.env_reset = env.reset
        self.EnvType.reset = self.reset
        self.env_step = env.step
        self.EnvType.step = self.step

        self.EnvType.get_episode_score = lambda _: self.score
        self.EnvType.get_episode_length = lambda _: self.eps_len

    def __call__(self, *args, **kwargs):
        self.env = self.EnvType(*args, **kwargs)

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

@ray.remote
class RayGymEnv:
    def __init__(self, name):
        self.env = gym.make(name)
        self.__dict__ = self.env.__dict__

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


@envstats
class GymEnv:
    def __init__(self, args):
        self.env = env = gym.make(args['name'])
        # if model_name:
        #     self.env = gym.wrappers.Monitor(self.env, f'data/gym/{model_name}', force=True)
        if 'atari' in args and args['atari']:
            env = atari_wrappers.wrap_deepmind(env)
        env.seed(args['seed'])

        self.state_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.max_episode_steps = args['max_episode_steps'] if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = np.squeeze(action)
        return self.env.step(action)
        
    def render(self):
        return self.env.render()

@envstats
class GymEnvVec:
    def __init__(self, args):
        assert 'n_envs' in args, colorize(f'Please specify n_envs in args.yaml beforehand', 'red')
        n_envs = args['n_envs']
        self.envs = [gym.make(args['name']) for i in range(n_envs)]
        [env.seed(args['seed'] + 10 * i) for i, env in enumerate(self.envs)]

        env = self.envs[0]
        self.state_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = env.action_space
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = n_envs
        self.max_episode_steps = args['max_episode_steps'] if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        actions = np.squeeze(actions)
        return list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))

