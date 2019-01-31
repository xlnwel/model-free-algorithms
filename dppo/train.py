import os
import time
from collections import deque
import logging
import numpy as np
import tensorflow as tf
import gym
import ray

from utils import yaml_op, logger
from worker import Worker
from learner import Learner

def get_model_name(args, name):
    option = args['option']
    option_str = ''
    for key, value in option.items():
        option_str = option_str + '{}-{}_'.format(key, value)
    model_name = option_str + name

    return model_name

def train(ppo_args, env_args):
    n_workers = ppo_args['n_workers']
    del ppo_args['n_workers']
    ppo_args['batch_size'] //= n_workers

    ppo_args['model_name'] = get_model_name(ppo_args, 'leaner')
    learner = Learner.remote('ppo', ppo_args, env_args, log_tensorboard=True, log_params=True, log_score=True)

    workers = []
    for i in range(1, n_workers+1):
        ppo_args['model_name'] = get_model_name(ppo_args, 'worker-{}'.format(i))
        workers.append(Worker.remote('ppo', ppo_args, env_args))
    
    print('Start Training...\n\n\n')
    
    weights_id = learner.get_weights.remote()
    score_deque = deque(maxlen=100)
    for i in range(1, 1000 + 1):
        score_ids = [worker.sample_trajectories.remote(weights_id) for worker in workers]

        for _ in range(ppo_args['n_updates_per_iteration']):
            grads_ids = [worker.compute_gradients.remote(weights_id) for worker in workers]
            
            weights_id = learner.apply_gradients.remote(*grads_ids)

        scores = ray.get(score_ids)
        score = np.mean(scores)
        score_deque.append(score)
        learner.log_score.remote(score, np.mean(score_deque))
        # logger.log_tabular('SamplingTime', sampling_time)
        logger.log_tabular('Iteration', i)
        logger.log_tabular('AverageScore', score)
        logger.log_tabular('StdScore', np.std(scores))
        logger.log_tabular('MaxScore', np.max(scores))
        logger.log_tabular('MinScore', np.min(scores))
        logger.dump_tabular()

def main():
    args = yaml_op.load_args()
    
    # setup logger
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = get_model_name(args['ppo'], 'data') +  '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)

    logger.configure_output_dir(logdir)
    logger.save_args(args)

    ray.init()
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    train(args['ppo'], args['env'])


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
