import os
import time
from collections import deque
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import gym
import ray

from utility import yaml_op
from algo.on_policy.ppo.agent import Agent


def main(env_args, agent_args, buffer_args, render=False):
    if 'n_workers' in agent_args:
        del agent_args['n_workers']

    agent = Agent('agent', agent_args, env_args)

    score_deque = deque(maxlen=100)
    eps_len_deque = deque(maxlen=100)
    for i in range(1, agent_args['n_epochs'] + 1):
        start = time.time()
        env_stats = agent.sample_trajectories()

        loss_info_list = []
        for _ in range(agent_args['n_updates']):
            if agent_args['shuffle']:
                assert agent_args['ac']['use_lstm'] == False, 'Should not shuffle data when using RNNs'
                agent.buffer.shuffle()
            for _ in range(agent_args['n_minibatches']):
                loss_info = agent.optimize()

                loss_info_list.append(loss_info)

        # score logging
        scores, eps_lens = env_stats
        score = np.mean(scores)
        eps_len = np.mean(eps_lens)
        score_deque.append(score)
        eps_len_deque.append(eps_len)
        
        # data logging
        loss_info = list(zip(*loss_info_list))
        ppo_loss, entropy, value_loss, total_loss, approx_kl, clip_frac = loss_info

        avg_score = np.mean(score_deque)
        approx_kl = np.mean(approx_kl)
        clip_frac = np.mean(clip_frac)
        agent.log_stats(score=score, avg_score=avg_score,
                        eps_len=eps_len, avg_eps_len=np.mean(eps_len_deque),
                        approx_kl=approx_kl, clip_frac=clip_frac)

        log_info = {
            'ModelName': agent_args['model_name'],
            'Iteration': i,
            'Time': f'{time.time() - start:3.2f}s',
            'AverageScore': score,
            'StdScore': np.std(scores),
            'MaxScore': np.max(scores),
            'MinScore': np.min(scores),
            'PPOLoss': np.mean(ppo_loss),
            'Entropy': np.mean(entropy),
            'ValueLoss': np.mean(value_loss),
            'TotalLoss': np.mean(total_loss),
            'ApproxKL': np.mean(approx_kl),
            'ClipFrac': np.mean(clip_frac)
        }
        [agent.log_tabular(k, v) for k, v in log_info.items()]
        agent.dump_tabular(print_terminal_info=True)
