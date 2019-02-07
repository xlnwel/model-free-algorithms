import os
import time
from collections import deque
from multiprocessing import Process
import logging
import numpy as np
import tensorflow as tf
import gym
import ray

from utility import yaml_op, logger
from worker import Worker
from learner import Learner


def get_model_name(args, name):
    # option = args['option']
    # option_str = ''
    # for key, value in option.items():
    #     option_str = option_str + '{}-{}_'.format(key, value)
    model_name = args['model_name'] + '_' + name

    return model_name

def train(agent_args, env_args):
    n_workers = agent_args['n_workers']
    del agent_args['n_workers']
    agent_args['minibatch_size'] //= n_workers

    agent_args['model_name'] = get_model_name(agent_args, 'leaner')
    learner = Learner.remote('agent', agent_args, env_args, 
                             log_tensorboard=True, 
                             log_params=True, 
                             log_score=True)

    workers = []
    for i in range(1, n_workers+1):
        agent_args['model_name'] = get_model_name(agent_args, 'worker-{}'.format(i))
        workers.append(Worker.remote('agent', agent_args, env_args))
    
    print('Start Training...\n\n\n')
    
    weights_id = learner.get_weights.remote()
    score_deque = deque(maxlen=100)
    for i in range(1, agent_args['n_epochs'] + 1):
        loss_info_list = []
        score_ids = [worker.sample_trajectories.remote(weights_id) for worker in workers]

        score_lists = ray.get(score_ids)
        
        for _ in range(agent_args['n_minibatches']):
            grads_ids = []
            losses_ids = []
            results = [worker.compute_gradients.remote(weights_id) for worker in workers]
            for r in results:
                grads_ids.append(r[0])
                losses_ids.append(r[1])
            loss_info_list += ray.get(losses_ids)

            weights_id = learner.apply_gradients.remote(*grads_ids)

        # score logging
        
        scores = []
        for sl in score_lists:
            scores += sl
        score = np.mean(scores)
        score_deque.append(score)
        learner.log_score.remote(score, np.mean(score_deque))
        
        # data logging
        actor_ppo_losses, actor_entropy, actor_losses, critic_losses, total_losses = [], [], [], [], []
        for losses in loss_info_list:
            assert len(losses) == 5, losses
            actor_ppo_losses.append(losses[0])
            actor_entropy.append(losses[1])
            actor_losses.append(losses[2])
            critic_losses.append(losses[3])
            total_losses.append(losses[4])

        logger.log_tabular('Iteration', i)
        logger.log_tabular('AverageScore', score)
        logger.log_tabular('StdScore', np.std(scores))
        logger.log_tabular('MaxScore', np.max(scores))
        logger.log_tabular('MinScore', np.min(scores))
        logger.log_tabular('ActorPPOLoss', np.mean(actor_ppo_losses))
        logger.log_tabular('ActorEntropy', np.mean(actor_entropy))
        logger.log_tabular('ActorLoss', np.mean(actor_losses))
        logger.log_tabular('CriticLoss', np.mean(critic_losses))
        logger.log_tabular('TotalLoss', np.mean(total_losses))
        logger.dump_tabular()

def train_agent(agent_args, env_args):
    # setup logger
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = agent_args['model_dir'] + '_'\
             + get_model_name(agent_args, 'data') +  '_'\
            + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)

    logger.configure_output_dir(logdir)
    logger.save_args(agent_args)

    # tf.logging.set_verbosity(tf.logging.ERROR)

    train(agent_args, env_args)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    ray.init(num_gpus=1)

    args = yaml_op.load_args()
    agent_args, env_args = args['agent'], args['env']
    
    if args['n_experiments'] == 1:
        train_agent(agent_args, env_args)
    else:
        processes = []

        for n_minibatches in [1, 3, 5]:
            agent_args['n_minibatches'] = n_minibatches
            agent_args['model_name'] = 'norm-minibatches_{}'.format(n_minibatches)

            p = Process(target=train_agent, args=(agent_args, env_args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
