"""
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
"""
import numpy as np
import gym

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EnvStats(gym.Wrapper):
    """ Provide Environment Stats Records """
    def reset(self):
        self.score = 0
        self.epslen = 0
        self.early_done = 0
        self.mask = 1
        
        return self.env.reset()

    def step(self, action):
        self.mask = 1 - self.early_done
        next_state, reward, done, info = self.env.step(action)
        self.score += 0 if self.early_done else reward
        self.epslen += 0 if self.early_done else 1
        self.early_done = done

        return next_state, reward, done, info

    def get_mask(self):
        """ Get mask at the current step. Should only be called after self.step """
        return self.mask

    def get_score(self):
        return self.score

    def get_epslen(self):
        return self.epslen

    @property
    def is_action_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def state_shape(self):
        return self.observation_space.shape

    @property
    def state_dtype(self):
        return self.observation_space.dtype

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def action_dtype(self):
        return np.int8 if self.is_action_discrete else (self.env.action_space.dtype)

    @property
    def action_dim(self):
        return self.action_space.n if self.is_action_discrete else self.env.action_space.shape[0]


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
            