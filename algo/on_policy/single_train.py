import os
import time
import numpy as np

from utility import utils
from algo.on_policy.ppo.agent import Agent


def train(agent, agent_args, test_agent):
    for i in range(1, agent_args['n_epochs'] + 1):
        start = time.time()
        env_stats = agent.sample_trajectories()

        loss_info_list = agent.optimize(i)

        # score logging
        scores, eps_lens = env_stats
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        score_max = np.max(scores)
        score_min = np.min(scores)
        epslen_mean = np.mean(eps_lens)
        
        # data logging
        loss_info = list(zip(*loss_info_list))
        ppo_loss, entropy, approx_kl, clip_frac, value_loss = loss_info

        ppo_loss = np.mean(ppo_loss)
        entropy = np.mean(entropy)
        approx_kl = np.mean(approx_kl)
        clip_frac = np.mean(clip_frac)
        value_loss = np.mean(value_loss)
        agent.record_stats(score_mean=score_mean, score_std=score_std,
                           epslen_mean=epslen_mean, approx_kl=approx_kl, 
                           clip_frac=clip_frac)

        log_info = {
            'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
            'Iteration': i,
            'Time': f'{time.time() - start:3.2f}s',
            'ScoreMean': score_mean,
            'ScoreStd': score_std,
            'ScoreMax': score_max,
            'ScoreMin': score_min,
            'PPOLoss': ppo_loss,
            'ValueLoss': value_loss,
            'Entropy': entropy,
            'ApproxKL': approx_kl,
            'ClipFrac': clip_frac
        }
        [agent.log_tabular(k, v) for k, v in log_info.items()]
        agent.dump_tabular(print_terminal_info=True)

        if test_agent:
            test_agent.demonstrate()

def main(env_args, agent_args, buffer_args, render=False):
    utils.set_global_seed()

    agent_name = 'Agent'
    agent = Agent(agent_name, agent_args, env_args, 
                  save=False, log=True, log_tensorboard=False, 
                  log_params=False, log_stats=True, device='/gpu:0')

    test_agent = None
    if render:
        env_args['n_envs'] = 1  # run test agent in a single environment
        env_args['log_video'] = True
        test_agent = Agent(agent_name, agent_args, env_args, save=False, log_tensorboard=False, 
                            log_params=False, log_stats=False, device='/gpu:0', reuse=True, graph=agent.graph)

    train(agent, agent_args, test_agent)
