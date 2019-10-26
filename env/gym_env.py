""" Implementation of single process environment """
import numpy as np
import gym
import ray

from utility import tf_distributions
from utility.debug_tools import assert_colorize
from utility.utils import pwc, to_int
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
        self.max_episode_steps = args.setdefault('max_episode_steps', env.spec.max_episode_steps)

        # clip reward at done
        self.clip_reward = (args['clip_reward'] if 'clip_reward' in args 
                            and not isinstance(args['clip_reward'], str) else None)
        self.n_envs = 1

        if 'log_video' in args and args['log_video']:
            pwc(f'video will be logged at {args["video_path"]}', 'cyan')
            env = gym.wrappers.Monitor(TimeLimit(env, self.max_episode_steps), args['video_path'], force=True)

        self.env = env = EnvStats(env)
        seed = args.setdefault('seed', 42)
        self.env.seed(seed)

        self.state_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
    def reset(self):
        return self.env.reset()

    def random_action(self):
        return self.env.action_space.sample()
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.clip_reward and done:
            reward = np.maximum(reward, self.clip_reward)

        return state, reward, done, info

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
        envs = [gym.make(args['name']) for i in range(n_envs)]
        [env.seed(args['seed'] + i) for i, env in enumerate(envs)]
        self.envs = [EnvStats(env) for env in envs]

        env = self.envs[0]
        self.state_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = env.action_space
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = n_envs
        self.max_episode_steps = to_int(args['max_episode_steps']) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps
        # clip reward at done
        self.clip_reward = (args['clip_reward'] if 'clip_reward' in args 
                            and not isinstance(args['clip_reward'], str) else None)

    def random_action(self):
        return [env.action_space.sample() for env in self.envs]

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        step_imp = lambda envs, actions: list(zip(*[env.step(a) for env, a in zip(envs, actions)]))
        
        state, reward, done, info = step_imp(self.envs, actions)
        masks = [env.get_mask() for env in self.envs]
        reward = np.where(masks, reward, 0)
        if self.clip_reward:
            reward = np.where(done, np.maximum(reward, self.clip_reward), reward)
        
        return state, reward, done, info

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return np.asarray([env.get_mask() for env in self.envs])

    def get_score(self):
        return np.asarray([env.get_score() for env in self.envs])

    def get_epslen(self):
        return np.asarray([env.get_epslen() for env in self.envs])


def create_env(args):
    if 'n_envs' not in args or args['n_envs'] == 1:
        return GymEnv(args)
    else:
        return GymEnvVec(args)
