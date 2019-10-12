"""
Code for training single agent
"""
import time
import threading
from collections import deque
import numpy as np
import tensorflow as tf

from utility import utils
from utility.tf_utils import get_sess_config
from utility.debug_tools import timeit


def run_trajectory(agent, fn, render, random_action=False):
    """ run a trajectory, fn is a function executed after each environment step """
    env = agent.env
    state = env.reset()
    for i in range(env.max_episode_steps):
        if render:
            env.render()
        action = env.random_action() if random_action else agent.act(state)
        next_state, reward, done, _ = env.step(action)
        fn(state, action, reward, done, i)
        state = next_state
        if done:
            break

    return env.get_score(), env.get_epslen()

def eval(agent, k, interval, render, print_terminal_info):
    def eval_fn(state, action, reward, done, i):
        pass    # do nothing at eval time

    scores, epslens = [], []

    for _ in range(1, interval+1):
        score, epslen = run_trajectory(agent, eval_fn, render)
        scores.append(score)
        epslens.append(epslen)
    utils.pwc(f'{agent.args["model_name"]} Evaluation after {k} training epochs\n'
              f'Average score: {np.mean(scores)}\n'
              f'Average episode length: {np.mean(epslens)}')

def train(agent, n_epochs, render, print_terminal_info):
    def pre_collection(state, action, reward, done, i):
        agent.add_data(state, action, reward, done)
    def train_fn(state, action, reward, done, i):
        agent.add_data(state, action, reward, done)
        if agent.buffer.good_to_learn and i % agent.args['update_freq'] == 0:
            agent.learn()

    interval = 100
    scores = deque(maxlen=interval)
    epslens = deque(maxlen=interval)

    t = 0

    utils.pwc(f'Data collection for state normalization')
    while not agent.buffer.good_to_learn:
        run_trajectory(agent, pre_collection, False, True)
    assert agent.buffer.good_to_learn
    utils.pwc(f'Training starts')

    for k in range(1, n_epochs + 1):
        duration, (score, epslen) = timeit(run_trajectory, (agent, train_fn, False))
        t += duration

        scores.append(score)
        epslens.append(epslen)

        if k % 10 == 0:
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            epslen_mean = np.mean(epslens)
            epslen_std = np.std(epslens)

            if hasattr(agent, 'stats'):
                agent.record_stats(score_mean=score_mean, score_std=score_std,
                                    epslen_mean=epslen_mean, epslen_std=epslen_std,
                                    global_step=k)
            
            log_info = {
                'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
                'Iteration': k,
                'Time': utils.timeformat(t) + 's',
                'ScoreMean': score_mean,
                'ScoreStd': score_std,
                'EpsLenMean': epslen_mean,
                'EpsLenStd': epslen_std
            }
            [agent.log_tabular(k, v) for k, v in log_info.items()]
            agent.dump_tabular(print_terminal_info=print_terminal_info)

        if k % 100 == 0:
            eval(agent, k, interval, render, print_terminal_info)

def main(env_args, agent_args, buffer_args, render=False):
    # print terminal information if main is running in the main thread
    utils.set_global_seed()

    algorithm = agent_args['algorithm']
    if algorithm == 'td3':
        from algo.off_policy.td3.agent import Agent
    elif algorithm == 'sac':
        from algo.off_policy.sac.agent import Agent
    elif algorithm == 'rainbow-iqn':
        from algo.off_policy.rainbow_iqn.agent import Agent
    else:
        raise NotImplementedError

    agent_args['env_stats']['times'] = 1
    sess_config = get_sess_config(1)

    agent = Agent('Agent', agent_args, env_args, buffer_args, 
                  sess_config=sess_config, log=True,
                  log_tensorboard=True, log_stats=True, 
                  save=True, device='/GPU: 0')

    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, agent_args['n_epochs'], render, 
          print_terminal_info=True)
