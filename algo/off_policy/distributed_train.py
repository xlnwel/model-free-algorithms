import os

import time
import argparse
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import ray

from utility.tf_utils import get_sess_config
from algo.off_policy.replay.proportional_replay import ProportionalPrioritizedReplay
from algo.off_policy.apex.worker import get_worker
from algo.off_policy.apex.learner import get_learner


def main(env_args, agent_args, buffer_args, render=False):
    if agent_args['algorithm'] == 'apex-td3':
        from algo.off_policy.td3.agent import Agent
    elif agent_args['algorithm'] == 'apex-sac':
        from algo.off_policy.sac.agent import Agent
    else:
        raise NotImplementedError

    if 'n_workers' not in agent_args:
        # 1 cpu for each actor
        n_workers = cpu_count() - 2
        agent_args['n_workers'] = n_workers
    else:
        n_workers = agent_args['n_workers']
    agent_args['env_stats']['times'] = n_workers

    ray.init()

    agent_name = 'Agent'
    sess_config = get_sess_config(2)
    learner = get_learner(Agent, agent_name, agent_args, env_args, buffer_args, 
                            log=True, log_tensorboard=True, log_stats=True, 
                            sess_config=sess_config, device='/GPU: 0')

    workers = []
    buffer_args['type'] = 'local'
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                 inter_op_parallelism_threads=1,
                                 allow_soft_placement=True)
    for worker_no in range(n_workers):
        weight_update_freq = 1    # np.random.randint(1, 10)
        if agent_args['algorithm'] == 'apex-td3':
            agent_args['actor']['noisy_sigma'] = 0.3 if worker_no == 0 else np.random.randint(4, 10) * .1
        elif agent_args['algorithm'] == 'apex-sac':
            agent_args['Policy']['noisy_sigma'] = 0.3 if worker_no == 0 else np.random.randint(4, 10) * .1
        else:
            raise NotImplementedError
        env_args['seed'] = worker_no * 10
        if render:
            if worker_no == 0:
                env_args['log_video'] = True
            else:
                env_args['log_video'] = False
        worker = get_worker(Agent, agent_name, worker_no, agent_args, env_args, buffer_args, 
                            weight_update_freq, sess_config=sess_config, device=f'/CPU:0')
        workers.append(worker)

    pids = [worker.sample_data.remote(learner) for worker in workers]

    ray.get(pids)
