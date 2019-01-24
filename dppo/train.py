import os
import time
import logging
import numpy as np
import tensorflow as tf
import gym
import ray

from utils import yaml_op, logger
from worker import Worker
from learner import Learner


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
        # start = time.time()
        score_ids = [worker.sample_trajectories.remote(weights_id) for worker in workers]
        
        # sampling_time = time.time() - start

        for _ in range(ppo_args['n_updates_per_iteration']):
            grads_ids = [worker.compute_gradients.remote(weights_id) for worker in workers]

            weights_id = learner.apply_gradients.remote(*grads_ids)

        scores = ray.get(score_ids)
        # logger.log_tabular('SamplingTime', sampling_time)
        logger.log_tabular('Iteration', i)
        logger.log_tabular('AverageScore', np.mean(scores))
        logger.log_tabular('StdScore', np.std(scores))
        logger.log_tabular('MaxScore', np.max(scores))
        logger.log_tabular('MinScore', np.min(scores))
        logger.dump_tabular()

        # al_id, cl_id = workers[0].compute_loss.remote()
        # print('Actor loss:{:>10.2f}, Critic loss:{:>10.2f}'.format(ray.get(al_id), ray.get(cl_id)))

def main():
    args = yaml_op.load_args()
    
    # setup logger
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args['ppo']['model_name'] + '_' + args['env']['name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)

    logger.configure_output_dir(logdir)
    logger.save_args(args)

    ray.init()
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    train(args['ppo'], args['env'])


if __name__ == '__main__':
    main()
