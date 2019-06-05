import os

import time
import argparse
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import ray

from utility.yaml_op import load_args
from replay.proportional_replay import ProportionalPrioritizedReplay
from algo.off_policy.apex.worker import get_worker
from algo.off_policy.apex.learner import get_learner


def main(env_args, agent_args, buffer_args, render=False):
    if agent_args['algorithm'] == 'td3':
        from algo.off_policy.td3.agent import Agent
    elif agent_args['algorithm'] == 'sac':
        from algo.off_policy.sac.agent import Agent
    else:
        raise NotImplementedError

    if 'n_workers' not in agent_args:
        agent_args['n_workers'] = cpu_count() - 2

    ray.init(num_cpus=agent_args['n_workers'] + 2, num_gpus=1)

    agent_name = 'Agent'
    learner = get_learner(Agent, agent_name, agent_args, env_args, buffer_args, device='/gpu: 0')

    workers = []
    buffer_args['type'] = 'local'
    for worker_no in range(agent_args['n_workers']):
        max_episodes = 1    # np.random.randint(1, 10)
        if agent_args['algorithm'] == 'td3':
            agent_args['actor']['noisy_sigma'] = np.random.randint(3, 7) * .1
        elif agent_args['algorithm'] == 'sac':
            agent_args['policy']['noisy_sigma'] = np.random.randint(3, 7) * .1
        else:
             raise NotImplementedError
        env_args['seed'] = worker_no * 10
        worker = get_worker(Agent, agent_name, worker_no, agent_args, env_args, buffer_args, 
                            max_episodes, device='/cpu: {}'.format(worker_no + 1))
        workers.append(worker)

    pids = [worker.sample_data.remote(learner) for worker in workers]

    ray.get(pids)
