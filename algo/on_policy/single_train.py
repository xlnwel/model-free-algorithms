import os
import time
from collections import deque
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import gym
import ray

from utility import yaml_op
from ppo.agent import Agent
from env.gym_env import GymEnvVec

def main(env_args, agent_args, buffer_args, render=False):
    if 'n_workers' in agent_args:
        del agent_args['n_workers']

    agent = Agent('agent', agent_args, env_args)

    score_deque = deque(maxlen=100)
    eps_len_deque = deque(maxlen=100)
    for i in range(1, agent_args['n_epochs'] + 1):
        start = time.time()
        env_stats = agent.sample_trajectories()

        if agent_args['advantage_type'] == 'gae':
            advs = agent.get_advantages()
            adv_mean = np.mean(advs)
            adv_std = np.std(advs)
            agent.normalize_advantages(adv_mean, adv_std)

        loss_info_list = []
        for _ in range(agent_args['n_updates']):
            for _ in range(agent_args['n_minibatches']):
                loss_info = agent.optimize()

                loss_info_list.append(loss_info)

        # score logging
        scores, eps_lens = env_stats
        score = np.mean(scores)
        eps_len = np.mean(eps_lens)
        score_deque.append(score)
        eps_len_deque.append(eps_len)
        logs_ids = [agent.log_stats(score=score, avg_score=np.mean(score_deque),
                                      eps_len=eps_len, avg_eps_len=np.mean(eps_len_deque))]
        
        # data logging
        loss_info = list(zip(*loss_info_list))
        ppo_loss, entropy, value_loss, total_loss, approx_kl, clipfrac = loss_info

        log_info = {
            'model_name': agent_args['model_name'],
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
            'ClipFrac': np.mean(clipfrac)
        }
        logs_ids += [agent.log_tabular(k, v) for k, v in log_info.items()]
        logs_ids.append(agent.dump_tabular(print_terminal_info=True))
