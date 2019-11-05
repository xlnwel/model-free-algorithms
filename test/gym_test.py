import random
import numpy as np

from env.gym_env import create_gym_env


args = dict(
    name='BipedalWalker-v2',
    video_path='video',
    log_video=False,
    max_episode_steps=1000,
    clip_reward=None,
    seed=0
)

class TestClass:
    def test_GymEnv(self):
        args['n_envs'] = 1
        env = create_gym_env(args)
        d = False
        cr = 0
        n = 0
        s = env.reset()
        while not d:
            n += 1
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r

        assert cr == env.get_score()
        assert n == env.get_epslen()
        return cr, n

    def test_GymEnvVec(self):
        args['n_envs'] = random.randint(5, 10)
        print(args['n_envs'])
        env = create_gym_env(args)
        d = False
        cr = np.zeros(env.n_envs)
        n = np.zeros(env.n_envs)
        s = env.reset()
        for _ in range(env.max_episode_steps):
            a = env.random_action()
            s, r, d, _ = env.step(a)
            cr += r
            n += np.where(env.get_mask(), 1, 0)
        assert np.all(cr == env.get_score()), f'counted reward: {cr}\nrecorded reward: {env.get_score()}'
        assert np.all(n == env.get_epslen()), f'counted epslen: {n}\nrecorded epslen: {env.get_epslen()}'
            
        return cr, n
