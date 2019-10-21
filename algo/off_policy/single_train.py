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
from algo.off_policy.apex.buffer import LocalBuffer


def evaluate(agent, timestep, start_episodes, interval, scores, epslens, render):
    for i in range(1, interval+1):
        score, epslen = agent.run_trajectory(render=render, deterministic_action=True)
        timestep += epslen
        scores.append(score)
        epslens.append(epslen)
        if i % 4 == 0:
            agent.rl_log(dict(Timing='Eval', 
                            Episodes=start_episodes+i,
                            Steps=timestep,
                            Score=score, 
                            ScoreMean=np.mean(scores),
                            ScoreStd=np.std(scores),
                            EpsLenMean=np.mean(epslens),
                            EpsLenStd=np.std(epslens)))
    return timestep

def train(agent, buffer, n_epochs, render):
    def collection_fn(state, action, reward, done, n):
        buffer.add_data(state, action, reward, done)

    def train_fn(state, action, reward, done, n):
        agent.add_data(state, action, reward, done)
        if agent.buffer.good_to_learn:
            for _ in range(n):
                agent.learn()

    def collect_data(agent, buffer, random_action=False):
        if buffer:
            buffer.reset()
            score, epslen = agent.run_trajectory(fn=collection_fn, random_action=random_action)
            buffer['priority'][:] = agent.buffer.top_priority
            agent.buffer.merge(buffer, buffer.idx)
        else:
            score, epslen = agent.run_trajectory(fn=train_fn, random_action=random_action)

        return score, epslen

    interval = 100
    train_timestep = 0
    scores = deque(maxlen=interval)
    epslens = deque(maxlen=interval)
    eval_interval = 50
    eval_timestep = 0
    test_scores = deque(maxlen=interval)
    test_epslens = deque(maxlen=interval)

    utils.pwc(f'Initialize replay buffer')
    while not agent.buffer.good_to_learn:
        collect_data(agent, buffer, random_action=True)
    assert agent.buffer.good_to_learn
    
    utils.pwc(f'Training starts')
    for episode_i in range(1, n_epochs + 1):
        score, epslen = collect_data(agent, buffer)
        train_timestep += epslen

        if buffer:
            for _ in range(epslen):
                agent.learn()

        scores.append(score)
        epslens.append(epslen)

        if episode_i % 4 == 0:
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            epslen_mean = np.mean(epslens)
            epslen_std = np.std(epslens)

            if hasattr(agent, 'stats'):
                agent.record_stats(score=score, score_mean=score_mean, score_std=score_std,
                                    epslen_mean=epslen_mean, epslen_std=epslen_std,
                                    global_step=episode_i)
            
            if hasattr(agent, 'logger'):
                agent.rl_log(dict(Timing='Train', 
                                Episodes=episode_i,
                                Steps=train_timestep,
                                Score=score, 
                                ScoreMean=score_mean,
                                ScoreStd=score_std,
                                EpsLenMean=epslen_mean,
                                EpsLenStd=epslen_std))

        if episode_i % eval_interval == 0:
            eval_timestep = evaluate(agent, eval_timestep, episode_i - eval_interval, 
                                    eval_interval, test_scores, test_epslens, render)

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

    if agent_args['episodic_learning']:
        # local buffer, only used to store a single episode of transitions
        buffer_args['local_capacity'] = env_args['max_episode_steps']
        buffer = LocalBuffer(buffer_args, agent.state_space, agent.action_dim)
    else:
        buffer = None

    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, buffer, agent_args['n_epochs'], render)
