import os
import gym
import random
import numpy as np
from collections import deque
import logging
from pathlib import Path
from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf
from multiprocessing import Process

from utils.debug_tools import timeit
import utils.utils as utils
from ddpg import DDPG

def print_args(args, i=0):
    for key, value in args.items():
        print('\t' * i + key, end=': ')
        if isinstance(value, dict):
            print()
            print_args(value, i+1)
        else:
            print(value)

def setup_logging(ddpg_args):
    log_dir = Path(ddpg_args['model_root_dir']) / ddpg_args['model_dir']
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True)

    log_filename = ddpg_args['model_name'] + '_timeit.log'
    logging.basicConfig(filename=str(log_dir / log_filename), level=logging.DEBUG)
    logging.info('gamma: {}, tau: {}'.format(ddpg_args['gamma'], ddpg_args['tau']))
    logging.info('actor\nlearning_rate: {}, noisy_sigma: {}'.format(
        ddpg_args['actor']['learning_rate'], ddpg_args['actor']['noisy_sigma']))
    logging.info('critic\nlearning_rate: {}'.format(
        ddpg_args['critic']['learning_rate']))
    logging.info('buffer\nalpha: {}, beta0: {}, beta_steps: {}'.format(
        ddpg_args['buffer']['alpha'], ddpg_args['buffer']['beta0'], ddpg_args['buffer']['beta_steps']))

def run_episodes(env, agent, n_episodes, scores_deque, hundred_episodes, 
                 on_notebook, image=None, print_terminal_info=False):
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        
        total_act_time = 0
        total_env_time = 0
        total_learn_time = 0
        for t in range(1000):
            if on_notebook:
                image.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                env.render()
            act_time, action = timeit(lambda: agent.act(state))
            env_time, (next_state, reward, done, _) = timeit(lambda: env.step(action))
            learn_time, _ = timeit(lambda: agent.learn(state, action, reward, next_state, done))
            total_act_time += act_time
            total_env_time += env_time
            total_learn_time += learn_time

            state = next_state
            score += reward
            if done:
                break
        avg_act_time = total_act_time / t
        avg_env_time = total_env_time / t
        avg_learn_time = total_learn_time / t
        logging.debug('Average act time at {} episodes: {:.2f} msec'.format(
            100 * hundred_episodes + i_episode, 1000 * avg_act_time))
        logging.debug('Average env time at {} episodes: {:.2f} msec'.format(
            100 * hundred_episodes + i_episode, 1000 * avg_env_time))
        logging.debug('Average learn time at {} episodes: {:.2f} msec'.format(
            100 * hundred_episodes + i_episode, 1000 * avg_learn_time))

        scores_deque.append(score)
        average_score = np.mean(scores_deque)
        agent.log_score(score, average_score)
        title = 'Training' if agent.trainable else 'Testing'
        if print_terminal_info:
            print('\r{}:\tEpisode {}\tAverage Score: {:3.2f}\tScore: {:3.2f}'.format(title, i_episode, average_score, score), end="")
    if print_terminal_info:
        print('\r{}:\tEpisode {}\tAverage Score: {:3.2f}'.format(title, i_episode, average_score))

def run(env, agent, on_notebook, n_episodes=3000, print_terminal_info=False):
    interval = 100
    scores_deque = deque(maxlen=interval)
    if on_notebook:
        image = plt.imshow(env.render(mode='rgb_array'))
    else:
        image = None
    
    for i in range(n_episodes // interval):
        if print_terminal_info:
            print('Start Episode {}'.format(i * interval))
        run_episodes(env, agent, interval, scores_deque, i, on_notebook, 
                     image, print_terminal_info=print_terminal_info)

def train(env_args, ddpg_args, on_notebook=False, print_terminal_info=False):
    if print_terminal_info:
        print('Arguments:')
        print_args(ddpg_args)
    
    # os.environ['PYTHONHASHSEED']=str(42)
    # random.seed(42)
    # np.random.seed(42)
    # tf.set_random_seed(42)

    env = gym.make(env_args['name'])
    if 'seed' in env_args:
        env.seed(env_args['seed'])
    
    ddpg_args['state_size'] = env.observation_space.shape[0]
    ddpg_args['action_size'] = env.action_space.shape[0]

    setup_logging(ddpg_args)

    agent_name = 'DDPG'

    agent = DDPG(agent_name, ddpg_args)
    agent.restore()
    print('Model {} starts training'.format(Path(ddpg_args['model_dir']) / ddpg_args['model_name']))
    
    run(env, agent, on_notebook, print_terminal_info=print_terminal_info)
    sess.close()

if __name__ == '__main__':
    args = utils.load_args()
    env_args = args['env']
    ddpg_args = args['ddpg']

    # disable tensorflow debug information
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    if args['n_experiments'] == 1:
        train(env_args, ddpg_args, print_terminal_info=True)
    else:
        processes = []

        for gamma in [.99]:
            for tau in [1e-3]:
                for noisy_sigma in [.4]:
                    for loss in ['huber', 'mse']:
                        for t in [1, 2]:
                            ddpg_args['gamma'] = gamma
                            ddpg_args['tau'] = tau
                            ddpg_args['actor']['noisy_sigma'] = noisy_sigma
                            ddpg_args['critic']['loss_type'] = loss
                            ddpg_args['model_dir'] = 'test'
                            ddpg_args['model_name'] = 'loss-{}_trial-{}'.format(loss, t)
                            p = Process(target=train, args=(env_args, ddpg_args))
                            p.start()
                            processes.append(p)
        
        for p in processes:
            p.join()
