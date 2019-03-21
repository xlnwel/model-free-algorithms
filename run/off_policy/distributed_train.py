import time
import argparse
import numpy as np
import tensorflow as tf
import ray

from utility.yaml_op import load_args
from replay.proportional_replay import ProportionalPrioritizedReplay
from td3.learner import Learner
from td3.worker import Worker


def main(env_args, agent_args, buffer_args, render=False):
    if agent_args['algorithm'] == 'td3':
        from td3.learner import Learner
        from td3.worker import Worker
    elif agent_args['algorithm'] == 'sac':
        from sac.learner import Learner
        from sac.worker import Worker
    else:
        raise NotImplementedError

    ray.init(num_cpus=agent_args['num_workers'] + 2, num_gpus=1)

    agent_name = 'Agent'
    learner = Learner.remote(agent_name, agent_args, env_args, buffer_args, device='/gpu: 0')

    workers = []
    buffer_args['type'] = 'local'
    for worker_no in range(agent_args['num_workers']):
        max_episodes = 1
        # max_episodes = np.random.randint(1, 10)
        # agent_args['actor']['noisy_sigma'] = np.random.randint(3, 10) * .1
        worker = Worker.remote(agent_name, worker_no, agent_args, env_args, buffer_args, 
                                max_episodes, device='/cpu: {}'.format(worker_no + 1))
        workers.append(worker)

    pids = [worker.sample_data.remote(learner) for worker in workers]

    ray.get(pids)
