import time
import numpy as np
import tensorflow as tf
import gym
import ray

from utility.yaml_op import load_args
from replay.proportional_replay import ProportionalPrioritizedReplay
from td3_rainbow.learner import Learner
from td3_rainbow.worker import Worker


def main():
    args = load_args('args.yaml')
    env_args = args['env']
    agent_args = args['agent']
    buffer_args = args['buffer']

    ray.init(num_cpus=12, num_gpus=1)

    agent_name = 'Agent'
    learner = Learner.remote(agent_name, agent_args, env_args, buffer_args, device='/gpu: 0')

    workers = []
    buffer_args['type'] = 'local'
    for worker_no in range(agent_args['num_workers']):
        store_episodes = np.random.randint(1, 10)
        agent_args['actor']['noisy_sigma'] = np.random.randint(3, 8) * .1
        worker = Worker.remote(agent_name, worker_no, agent_args, env_args, buffer_args, 
                    store_episodes, device='/cpu: {}'.format(worker_no + 1))
        workers.append(worker)

    pids = [worker.sample_data.remote(learner) for worker in workers]

    ray.get(pids)
    

if __name__ == '__main__':
    main()

