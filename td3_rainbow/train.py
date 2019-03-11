"""
Code for training single agent
"""

import os
import gym
import random
import numpy as np
from collections import deque
import logging
import threading
from pathlib import Path
from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf
from multiprocessing import Process

from utility.debug_tools import timeit
from utility import utils, yaml_op
from gym_env.env import GymEnvironment
from replay.proportional_replay import ProportionalPrioritizedReplay
from agent import Agent


def print_args(args, i=0):
    for key, value in args.items():
        print('\t' * i + key, end=': ')
        if isinstance(value, dict):
            print()
            print_args(value, i+1)
        else:
            print(value)

def setup_logging(agent_args, buffer_args):
    log_dir = Path(agent_args['model_root_dir']) / agent_args['model_dir']
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True)

    log_filename = agent_args['model_name'] + '_timeit.log'
    logging.basicConfig(filename=str(log_dir / log_filename), level=logging.DEBUG)
    logging.info('gamma: {}, tau: {}'.format(agent_args['gamma'], agent_args['tau']))
    logging.info('actor\nlearning_rate: {}, noisy_sigma: {}'.format(
        agent_args['actor']['optimizer']['learning_rate'], agent_args['actor']['noisy_sigma']))
    logging.info('critic\nlearning_rate: {}'.format(
        agent_args['critic']['optimizer']['learning_rate']))
    logging.info('buffer\nalpha: {}, beta0: {}, beta_steps: {}'.format(
        buffer_args['alpha'], buffer_args['beta0'], buffer_args['beta_steps']))

def run_episodes(env, agent, n_episodes, scores_deque, hundred_episodes, 
                 on_notebook, image=None, print_terminal_info=False):
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        
        total_act_time = 0
        total_env_time = 0
        for t in range(1000):
            if on_notebook:
                image.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                env.render()
            act_time, action = timeit(lambda: agent.act(state))
            env_time, (next_state, reward, done, _) = timeit(lambda: env.step(action))
            agent.add_data(state, action, reward, next_state, done)
            total_act_time += act_time
            total_env_time += env_time

            state = next_state
            score += reward
            if done:
                break
        avg_act_time = total_act_time / t
        avg_env_time = total_env_time / t
        logging.debug('Average act time at {} episodes: {:.2f} msec'.format(
            100 * hundred_episodes + i_episode, 1000 * avg_act_time))
        logging.debug('Average env time at {} episodes: {:.2f} msec'.format(
            100 * hundred_episodes + i_episode, 1000 * avg_env_time))

        scores_deque.append(score)
        average_score = np.mean(scores_deque)
        agent.log_score(score, average_score)
        title = 'Training' if agent.trainable else 'Testing'
        if print_terminal_info:
            print('\r{}:\tEpisode {}\tAverage Score: {:3.2f}\tScore: {:3.2f}'.format(
                title, i_episode, average_score, score), end="")
    if print_terminal_info:
        print('\r{}:\tEpisode {}\tAverage Score: {:3.2f}'.format(title, i_episode, average_score))

def train(env, agent, on_notebook, n_episodes=3000, print_terminal_info=False):
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

def main(env_args, agent_args, buffer_args, on_notebook=False, print_terminal_info=False):
    if print_terminal_info:
        print('Agent Arguments:\n')
        print_args(agent_args)
        print('Buffer Arguments:\n')
        print_args(buffer_args)
    
    # os.environ['PYTHONHASHSEED']=str(42)
    # random.seed(42)
    # np.random.seed(42)
    # tf.set_random_seed(42)

    env = GymEnvironment(env_args['name'])
    setup_logging(agent_args, buffer_args)

    agent_name = 'Agent'
    
    agent = Agent(agent_name, agent_args, env_args, buffer_args, log_tensorboard=True, log_score=True, device='/gpu:0')
    lt = threading.Thread(target=agent.background_learning, args=())
    lt.start()
    print('Model {} starts training'.format(Path(agent_args['model_dir']) / agent_args['model_name']))
    
    train(env, agent, on_notebook, print_terminal_info=print_terminal_info)

if __name__ == '__main__':
    args = yaml_op.load_args()
    env_args = args['env']
    agent_args = args['agent']
    buffer_args = args['buffer']

    # disable tensorflow debug information
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    if args['n_experiments'] == 1:
        main(env_args, agent_args, buffer_args, print_terminal_info=True)
    else:
        processes = []

        for gamma in [.99]:
            for tau in [1e-3]:
                for noisy_sigma in [.4]:
                    for loss in ['huber', 'mse']:
                        for t in [1, 2]:
                            agent_args['gamma'] = gamma
                            agent_args['tau'] = tau
                            agent_args['actor']['noisy_sigma'] = noisy_sigma
                            agent_args['critic']['loss_type'] = loss
                            agent_args['model_dir'] = 'test'
                            agent_args['model_name'] = 'loss-{}_trial-{}'.format(loss, t)
                            p = Process(target=main, args=(env_args, agent_args, buffer_args))
                            p.start()
                            processes.append(p)
        
        for p in processes:
            p.join()
