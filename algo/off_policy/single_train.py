"""
Code for training single agent. The agent trains its networks after every "update_freq" steps.
"""
import time
import threading
from collections import deque
import numpy as np
import tensorflow as tf

from utility import utils
from utility.tf_utils import get_sess_config
from utility.debug_tools import timeit


def log(agent, **kwargs):
    assert 'Timing' in kwargs
    assert 'Iteration' in kwargs
    assert 'Score' in kwargs
    assert 'ScoreMean' in kwargs
    assert 'ScoreStd' in kwargs
    assert 'EpsLenMean' in kwargs
    assert 'EpsLenStd' in kwargs

    log_info = {
        'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}'
    }
    log_info.update(kwargs)

    [agent.log_tabular(k, v) for k, v in log_info.items()]
    agent.dump_tabular()
    
def run_trajectory(agent, fn, render=False, random_action=False, test=False):
    """ run a trajectory, fn is a function executed after each environment step """
    env = agent.env
    state = env.reset()
    for i in range(env.max_episode_steps):
        if render:
            env.render()
        action = env.random_action() if random_action else agent.act(state, deterministic=test)
        next_state, reward, done, _ = env.step(action)
        fn(state, action, reward, done, i)
        state = next_state
        if done:
            break

    return env.get_score(), env.get_length()

def eval(agent, k, interval, scores, epslens, render):
    def eval_fn(state, action, reward, done, i):
        pass    # do nothing at eval time

    for i in range(1, interval + 1):
        score, epslen = run_trajectory(agent, eval_fn, render=render, test=True)
        scores.append(score)
        epslens.append(epslen)
        if i % 4 == 0:
            log(agent, 
                Timing='Eval', 
                Iteration=k-100+i,
                Score=score, 
                ScoreMean=np.mean(scores),
                ScoreStd=np.std(scores),
                EpsLenMean=np.mean(epslens),
                EpsLenStd=np.std(epslens))

def train(agent, n_epochs, render):
    def collection_fn(state, action, reward, done, i):
        agent.add_data(state, action, reward, done)
        
    def train_fn(state, action, reward, done, i):
        agent.add_data(state, action, reward, done)
        if agent.buffer.good_to_learn and i % agent.args['update_freq'] == 0:
            agent.learn()

    interval = 100
    scores = deque(maxlen=interval)
    epslens = deque(maxlen=interval)
    test_scores = deque(maxlen=interval)
    test_epslens = deque(maxlen=interval)

    utils.pwc(f'Data collection for state normalization')
    while not agent.buffer.good_to_learn:
        run_trajectory(agent, collection_fn, random_action=True)
    assert agent.buffer.good_to_learn
    utils.pwc(f'Training starts')

    for k in range(1, n_epochs + 1):
        score, epslen = run_trajectory(agent, train_fn)

        scores.append(score)
        epslens.append(epslen)

        if k % 4 == 0:
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            epslen_mean = np.mean(epslens)
            epslen_std = np.std(epslens)

            if hasattr(agent, 'stats'):
                agent.record_stats(score=score, score_mean=score_mean, score_std=score_std,
                                    epslen_mean=epslen_mean, epslen_std=epslen_std,
                                    global_step=k)
            
            if hasattr(agent, 'logger'):
                t, _ = timeit(log, **dict(agent=agent, 
                                Timing='Train', 
                                Iteration=k,
                                Score=score, 
                                ScoreMean=score_mean,
                                ScoreStd=score_std,
                                EpsLenMean=epslen_mean,
                                EpsLenStd=epslen_std))

        if k % 100 == 0:
            eval(agent, k, interval, test_scores, test_epslens, render)

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
                  save=False, device='/GPU: 0')

    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, agent_args['n_epochs'], render)
