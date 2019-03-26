import os
import time
from collections import deque
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import gym
import ray

from utility import yaml_op
from ppo.a2c.worker import Worker
from ppo.a2c.learner import Learner
from utility.utils import colorize
from env.gym_env import GymEnvVec

def main(env_args, agent_args, buffer_args, render=False):
    n_workers = agent_args['n_workers']
    del agent_args['n_workers']
    agent_args['batch_size'] //= n_workers

    ray.init(num_cpus=n_workers+1, num_gpus=1)
    learner = Learner.remote('agent', agent_args, env_args, 
                             log_tensorboard=True, 
                             log_params=True, 
                             log_score=True)


    workers = [Worker.remote('agent', i, agent_args, env_args) for i in range(n_workers)]
    
    weights_id = learner.get_weights.remote()
    score_deque = deque(maxlen=100)
    for i in range(1, agent_args['n_epochs'] + 1):
        loss_info_list = []
        score_ids = [w.sample_trajectories.remote(weights_id) for w in workers]

        advs = ray.get([w.get_advantages.remote() for w in workers])
        scores = ray.get(list(score_ids))  # ray cannot get 
        print('Start doing gradient descent')
        for _ in range(agent_args['n_batches']):
            adv_mean = np.mean(advs)
            adv_std = np.std(advs)
            grads_ids, losses_ids = list(zip(*[w.compute_gradients.remote(weights_id, adv_mean, adv_std)
                                                for w in workers]))

            loss_info_list += ray.get(list(losses_ids))

            weights_id = learner.apply_gradients.remote(*grads_ids)

        # score logging
        score = np.mean(scores)
        score_deque.append(score)
        learner.log_score.remote(score, np.mean(score_deque))
        
        # data logging
        loss_info = list(zip(*loss_info_list))
        ppo_loss, entropy, value_loss, total_loss, approx_kl, clipfrac = loss_info

        log_info = {
            'Iteration': i,
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
        ids = [learner.log_tabular.remote(k, v) for k, v in log_info.items()]
        ids.append(learner.dump_tabular.remote(print_terminal_info=True))

        ray.get(ids)