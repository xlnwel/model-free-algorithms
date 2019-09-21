import os
import time
from collections import deque
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import gym
import ray

from utility import yaml_op
from utility.debug_tools import pwc
from algo.on_policy.a2c.worker import Worker
from algo.on_policy.a2c.learner import Learner


def main(env_args, agent_args, buffer_args, render=False):
    def decompose(info):
        """ convert info () """
        return list(zip(*info))

    n_workers = agent_args['n_workers']
    del agent_args['n_workers']     # do not remove this!

    ray.init(num_cpus=n_workers+1, num_gpus=1)

    sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                 inter_op_parallelism_threads=1,
                                 allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    learner = Learner.remote('Agent', agent_args, env_args, sess_config=sess_config, device='/gpu: 0')
    workers = [Worker.remote('Agent', i, agent_args, env_args, sess_config=sess_config, device='/gpu: 0') 
               for i in range(n_workers)]

    max_kl = agent_args['max_kl']
    
    weights_id = learner.get_weights.remote()
    for epoch_i in range(1, agent_args['n_epochs'] + 1):
        start = time.time()
        env_stats = [w.sample_trajectories.remote(weights_id) for w in workers]

        if agent_args['advantage_type'] == 'gae':
            advs = ray.get([w.get_advantages.remote() for w in workers])
            adv_mean = np.mean(advs)
            adv_std = np.std(advs)
            [w.normalize_advantages.remote(adv_mean, adv_std) for w in workers]

        loss_info_list = []
        kl = 0

        for i in range(agent_args['n_updates']):
            for j in range(agent_args['n_minibatches']):
                grads_ids, losses_ids = decompose([w.compute_gradients.remote(weights_id) for w in workers])

                loss_info = ray.get(list(losses_ids))
                loss_info_list += loss_info

                weights_id = learner.apply_gradients.remote(epoch_i, *grads_ids)

                # we do not check KL in early stage
                if epoch_i < 100 or max_kl == 0:
                    continue

                loss_info = decompose(loss_info)
                kl = np.mean(loss_info[4])
                if kl > max_kl:
                    pwc(f'a2c: Eearly stopping at epoch-{epoch_i} update-{i} minibatch-{j} due to reaching max kl.\nCurrent kl={kl}')
                    break
            if max_kl == 0:
                continue
            
            if kl > max_kl:
                break

        # data logging
        loss_info = decompose(loss_info_list)
        ppo_loss, entropy, value_loss, total_loss, approx_kl, clip_frac = loss_info

        env_stats = ray.get(list(env_stats))  # ray cannot use tuple as input
        scores, epslens = list(zip(*env_stats))
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        epslen_mean = np.mean(epslens)
        ppo_loss = np.mean(ppo_loss)
        entropy = np.mean(entropy)
        value_loss = np.mean(value_loss)
        total_loss = np.mean(total_loss)
        approx_kl = np.mean(approx_kl)
        clip_frac = np.mean(clip_frac)
        logs_ids = [learner.record_stats.remote(score_mean=score_mean, score_std=score_std,
                                                epslen_mean=epslen_mean, entropy=entropy,
                                                approx_kl=approx_kl, clip_frac=clip_frac)]

        log_info = {
            'Iteration': epoch_i,
            'Time': f'{time.time() - start:3.2f}s',
            'ScoreMean': score_mean,
            'ScoreStd': score_std,
            'PPOLoss': ppo_loss,
            'Entropy': entropy,
            'ValueLoss': value_loss,
            'TotalLoss': total_loss,
            'ApproxKL': approx_kl,
            'ClipFrac': clip_frac
        }
        logs_ids += [learner.log_tabular.remote(k, v) for k, v in log_info.items()]
        logs_ids.append(learner.dump_tabular.remote(print_terminal_info=True))

        ray.get(logs_ids)
