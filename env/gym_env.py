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
        # Monitor cannot be used when an episode is terminated due to reaching max_episode_steps
        if 'log_video' in args and args['log_video']:
            pwc(f'video will be logged at {args["video_path"]}')
            env = gym.wrappers.Monitor(TimeLimit(env, args['max_episode_steps']), args['video_path'], force=True)
        self.env = env = EnvStats(env)

        env.seed(('seed' in args and args['seed']) or 42)

        self.state_space = env.observation_space.shape

        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = 1
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps
        # number of times repeating an action
        self.n_action_repetition = args['n_action_repetition'] if 'n_action_repetition' in args else 1
        # clip reward at done
        self.clip_reward = (args['clip_reward'] if 'clip_reward' in args 
                            and not isinstance(args['clip_reward'], str) else None)

    def reset(self):
        return self.env.reset()

    def random_action(self):
        return self.env.action_space.sample()
        
    def step(self, action):
        action = np.squeeze(action)
        cumulative_reward = 0.
        for _ in range(self.n_action_repetition):
            state, reward, done, info = self.env.step(action)
            if self.clip_reward and done:
                reward = np.maximum(reward, self.clip_reward)
            cumulative_reward += reward
            if done:
                break
        return state, cumulative_reward, done, info

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
        self.envs = [EnvStats(env) for env in envs]
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
        # number of times repeating an action
        self.n_action_repetition = args['n_action_repetition'] if 'n_action_repetition' in args else 1
        # clip reward at done
        self.clip_reward = (args['clip_reward'] if 'clip_reward' in args 
                            and not isinstance(args['clip_reward'], str) else None)

    def random_action(self):
        return [env.action_space.sample() for env in self.envs]

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        actions = np.squeeze(actions)
        step_imp = lambda envs, actions: list(zip(*[env.step(a) for env, a in zip(envs, actions)]))
        
        cumulative_reward = np.zeros(self.n_envs)
        for _ in range(self.n_action_repetition):
            state, reward, done, info = step_imp(self.envs, actions)
            if self.clip_reward:
                reward = np.where(done, np.maximum(reward, self.clip_reward), reward)
            cumulative_reward += reward

        return state, cumulative_reward, done, info

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return [env.get_mask() for env in self.envs]

    def get_score(self):
        return [env.get_score() for env in self.envs]

    def get_epslen(self):
        return [env.get_epslen() for env in self.envs]


if __name__ == '__main__':
    def run_traj(env):
        d = False
        cr = np.squeeze(np.zeros(env.n_envs))
        s = env.reset()
        i = 0
        while not d:
            i += 1
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r

        return cr, i

    def run_vec_traj(env):
        d = False
        cr = np.squeeze(np.zeros(env.n_envs))
        s = env.reset()
        i = 0
        for _ in range(env.max_episode_steps):
            i += 1
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r
            
        return cr, i

    args = dict(
        name='BipedalWalker-v2',
        video_path='video',
        log_video=False,
        max_episode_steps=1000,
        n_action_repetition=1,
        clip_reward=-10,
        n_envs=3,
        seed=0
    )
    print('******GymEnv******')
    env = GymEnv(args)
    r, i = run_traj(env)
    print(f'cumulative reward: {r}\t, score:{env.get_score()}')
    print(f'record length: {i}\t, actual length:{env.get_epslen()}')
    print(f'mask(1): {env.get_mask()}')
    env.step(env.random_action())
    print(f'mask(0): {env.get_mask()}')
    print('******GymEnvVec******')
    env = GymEnvVec(args)
    r, i = run_vec_traj(env)
    print(f'cumulative reward: {r}\t, score:{env.get_score()}')
    print(f'record length: {i}\t, actual length:{env.get_epslen()}')
    print(f'mask(1): {env.get_mask()}')
    env.step(env.random_action())
    print(f'mask(0): {env.get_mask()}')
