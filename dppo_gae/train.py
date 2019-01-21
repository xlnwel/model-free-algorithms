from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf
import gym
import ray

from worker import Worker
from learner import Learner
from utils import utils

def train(ppo_args, env_args):
    def change_model_name(args, name):
        args['model_name'] = name

    n_workers = ppo_args['n_workers']
    del ppo_args['n_workers']
    ppo_args['batch_size'] //= n_workers

    change_model_name(ppo_args, 'leaner')
    learner = Learner.remote('ppo-gae', ppo_args, env_args, log_tensorboard=True)

    workers = []
    for i in range(1, n_workers+1):
        change_model_name(ppo_args, 'worker-{}'.format(i))
        workers.append(Worker.remote('ppo-gae', ppo_args, env_args))
    
    print('Start Training...\n\n\n')
    
    weights_id = learner.get_weights.remote()
    for i in range(1, 2000 + 1):
        score_ids = [worker.sample_trajectories.remote(weights_id) for worker in workers]
        print('Iteration-{:<5d}Average score:{:>5.2}'.format(i, np.mean(ray.get(score_ids))))
        
        for _ in range(ppo_args['n_updates_per_iteration']):
            grads_ids = [worker.compute_gradients.remote(weights_id) for worker in workers]
            
            weights_id = learner.apply_gradients.remote(*grads_ids)


if __name__ == '__main__':

    # ray.init(redis_address='192.168.1.105:4869')
    ray.init()
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = utils.load_args()
    train(args['ppo-gae'], args['env'])
