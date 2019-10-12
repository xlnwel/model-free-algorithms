import numpy as np
import gym
import ray

from utility import tf_distributions
from utility.debug_tools import assert_colorize
from utility.utils import pwc
from env.wrappers import TimeLimit, EnvStats

def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError


class GymEnv:
    def __init__(self, args):
        env = gym.make(args['name'])
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps
        # Monitor cannot be used when an episode is terminated due to reaching max_episode_steps
        if 'log_video' in args and args['log_video']:
            pwc(f'video will be logged at {args["video_path"]}', 'cyan')
            env = gym.wrappers.Monitor(TimeLimit(self.env, self.max_episode_steps), args['video_path'], force=True)
        self.env = env = EnvStats(env)
        env.seed(args['seed'] if 'seed' in args else 42)

        self.state_space = env.observation_space.shape

        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = 1
        self.n_action_repetition = args['n_action_repetition'] if 'n_action_repetition' in args else 1

    def reset(self):
        return self.env.reset()

    def random_action(self):
        return self.env.action_space.sample()
        
    def step(self, action):
        action = np.squeeze(action)
        cumulative_reward = 0
        for i in range(self.n_action_repetition):
            next_state, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            if done:
                break
        return next_state, cumulative_reward, done, info

    def render(self):
        return self.env.render()

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return self.env.get_mask()

    def get_score(self):
        return self.env.get_score()
    
    def get_epslen(self):
        return self.env.get_epslen()


class GymEnvVec:
    def __init__(self, args):
        assert_colorize('n_envs' in args, f'Please specify n_envs in args.yaml beforehand')
        n_envs = args['n_envs']
        self.envs = [gym.make(args['name']) for i in range(n_envs)]
        self.envs = [EnvStats(env) for env in self.envs]
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
        self.n_action_repetition = args['n_action_repetition'] if 'n_action_repetition' in args else 1

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        actions = np.squeeze(actions)
        step_imp = lambda actions: list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))

        cumulative_reward = np.zeros(actions.shape[0])
        for _ in range(self.n_action_repetition):
            next_state, reward, done, info = step_imp(actions)
            mask = self.get_mask()
            cumulative_reward += np.where(mask, reward, 0)

        return next_state, reward, done, info

    def get_mask(self):
        return [env.get_mask() for env in self.envs]

    def get_score(self):
        return [env.get_score() for env in self.envs]

    def get_epslen(self):
        return [env.get_epslen() for env in self.envs]
