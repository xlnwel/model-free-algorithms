"""
Code for training single agent
"""

import os
from time import time
import random
import argparse
from collections import deque
import logging
import threading
import multiprocessing
from pathlib import Path
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utility import utils, yaml_op
from run.grid_search import GridSearch


def set_global_seed():
    os.environ['PYTHONHASHSEED']=str(42)
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm',
                        type=str,
                        choices=['td3', 'sac'])
    parser.add_argument('--render',
                        type=str,
                        choices=['true', 'false'],
                        default='false')
    parser.add_argument('--trials',
                        type=int,
                        default=1)
    args = parser.parse_args()

    return args

def print_args(args, i=1):
    for key, value in args.items():
        print('\t' * i + key, end=': ')
        if isinstance(value, dict):
            print()
            print_args(value, i+1)
        else:
            print(value)

def setup_logging(agent_args):
    log_dir = Path(agent_args['tensorboard_root_dir']) / agent_args['model_dir']
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = agent_args['model_name'] + '.log'
    logging.basicConfig(filename=str(log_dir / log_filename), level=logging.DEBUG)

def log_args(args, i=0):
    for key, value in args.items():
        logging.info('\t' * i + key, end=': ')
        if isinstance(value, dict):
            logging.info('\n')
            log_args(value, i+1)
        else:
            logging.info(value)


def run_episodes(agent, n_episodes, scores_deque, hundred_episodes, 
                 render=False, print_terminal_info=False):
    start = time()
    for episode_i in range(1, n_episodes+1):
        state = agent.env.reset()
        score = 0

        for _ in range(1000):
            if render:
                agent.env.render()
            action = agent.act(state)
            next_state, reward, done, _ = agent.env.step(action)
            # if not (np.any(np.isnan(state) | np.isinf(state)) or np.any(np.isnan(next_state) | np.isinf(next_state))):
            agent.add_data(state, action, reward, next_state, done)
            # if agent.buffer.good_to_learn:
            #     agent.learn()
            state = next_state
            score += reward
            if done:
                break

        scores_deque.append(score)
        average_score = np.mean(scores_deque)
        agent.log_score(score, average_score)
        title = 'Training' if agent.trainable else 'Testing'

        if print_terminal_info:
            print(f'\r{title}:\tEpisode {episode_i}\tAverage Score: {average_score:3.2f}\tScore: {score:3.2f}', end="")

    print(f'Model {agent.model_name} takes {(time() - start)/100:3.2f} seconds in average to run an episode.')
    if print_terminal_info:
        print(f'\r{title}:\tEpisode {episode_i}\tAverage Score: {average_score:3.2f}')

def train(agent, render, n_episodes=3000, print_terminal_info=False):
    interval = 100
    scores_deque = deque(maxlen=interval)
    
    for i in range(n_episodes // interval):
        if print_terminal_info:
            print(f'Start Episode {i * interval}')
        run_episodes(agent, interval, scores_deque, i, render, 
                     print_terminal_info=print_terminal_info)

def main(env_args, agent_args, buffer_args, render=False):
    # print terminal information if main is running in the main thread
    print_terminal_info = multiprocessing.current_process() == 'MainProcess'
    if print_terminal_info:
        print('Agent Arguments:')
        print_args(agent_args)
        print('Buffer Arguments:')
        print_args(buffer_args)
    
    set_global_seed()

    setup_logging(agent_args)
    log_args(env_args)
    log_args(agent_args)
    log_args(buffer_args)

    agent_name = 'Agent'
    algorithm = agent_args['algorithm']
    if algorithm == 'td3':
        from td3.agent import Agent
    elif algorithm == 'sac':
        from sac.agent import Agent
    else:
        raise NotImplementedError

    agent = Agent(agent_name, agent_args, env_args, buffer_args, log_tensorboard=True, log_score=True, device='/gpu:0')
    lt = threading.Thread(target=agent.background_learning, daemon=True)
    lt.start()
    model = Path(agent_args['model_dir']) / agent_args['model_name']
    print(f'Model {model} starts training')
    
    train(agent, render, print_terminal_info=print_terminal_info)

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = cmd_args.algorithm

    if algorithm == 'td3':
        arg_file = 'td3/args.yaml'
    elif algorithm == 'sac':
        arg_file = 'sac/args.yaml'
    else:
        raise NotImplementedError

    # disable tensorflow debug information
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    render = True if cmd_args.render == 'true' else False

    gs = GridSearch(arg_file, main, render, n_trials=cmd_args.trials)

    # Grid search happens here
    critic_args = {'learning_rate': [1e-4, 3e-4]}
    gs(policy_delay=[1, 2], critic=critic_args)
