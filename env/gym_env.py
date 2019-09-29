import numpy as np
import gym
import ray

from utility import tf_distributions
from utility.debug_tools import assert_colorize
from utility.utils import pwc
from env.wrappers import TimeLimit

def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError

class envstats:
    """ Provide Environment Stats Records """
    def __init__(self, env_type):
        self.EnvType = env_type
        self.score = 0
        self.eps_len = 0
        self.early_done = 0
        self.mask = 0
        
        self.env_reset = env_type.reset
        self.EnvType.reset = self.reset
        self.env_step = env_type.step
        self.EnvType.step = self.step
        self.EnvType.get_mask = self.get_mask

        self.EnvType.get_score = lambda _: self.score
        self.EnvType.get_length = lambda _: self.eps_len

    def __call__(self, *args, **kwargs):
        self.env = self.EnvType(*args, **kwargs)

        return self.env

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return self.mask

    def reset(self):
        self.score = 0
        self.eps_len = 0
        self.early_done = np.squeeze(np.zeros(self.env.n_envs))
        self.mask = np.squeeze(np.ones(self.env.n_envs))
        
        return self.env_reset(self.env)

    def step(self, action):
        self.mask = 1 - self.early_done
        next_state, reward, done, info = self.env_step(self.env, action)
        self.score += np.where(self.early_done, 0, reward)
        self.eps_len += np.where(self.early_done, 0, 1)
        self.early_done = np.array(done)

        return next_state, reward, done, info

@envstats
class GymEnv:
    def __init__(self, args):
        self.env = env = gym.make(args['name'])
        # Monitor cannot be used when an episode is terminated due to reaching max_episode_steps
        if 'log_video' in args and args['log_video']:
            pwc(f'video will be logged at {args["video_path"]}')
            self.env = env = gym.wrappers.Monitor(TimeLimit(self.env, args['max_episode_steps']), args['video_path'], force=True)

        env.seed(('seed' in args and args['seed']) or 42)

        self.state_space = env.observation_space.shape

        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = 1
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def reset(self):
        return self.env.reset()

    def random_action(self):
        return self.env.action_space.sample()
        
    def step(self, action):
        action = np.squeeze(action)
        return self.env.step(action)

    def render(self):
        return self.env.render()

@envstats
class GymEnvVec:
    def __init__(self, args):
        assert_colorize('n_envs' in args, f'Please specify n_envs in args.yaml beforehand')
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
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        actions = np.squeeze(actions)
        return list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
