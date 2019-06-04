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
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        avg_eps_len = np.mean(eps_lens)
        
        # data logging
        loss_info = list(zip(*loss_info_list))
        ppo_loss, entropy, value_loss, total_loss, approx_kl, clip_frac = loss_info

        approx_kl = np.mean(approx_kl)
        clip_frac = np.mean(clip_frac)
        agent.log_stats(avg_score=avg_score, std_score=std_score, max_score=max_score, min_score=min_score,
                        avg_eps_len=avg_eps_len, approx_kl=approx_kl, clip_frac=clip_frac)

        log_info = {
            'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
            'Iteration': i,
            'Time': f'{time.time() - start:3.2f}s',
            'AvgScore': avg_score,
            'StdScore': std_score,
            'MaxScore': max_score,
            'MinScore': min_score,
            'PPOLoss': np.mean(ppo_loss),
            'Entropy': np.mean(entropy),
            'ValueLoss': np.mean(value_loss),
            'TotalLoss': np.mean(total_loss),
            'ApproxKL': np.mean(approx_kl),
            'ClipFrac': np.mean(clip_frac)
        }
        [agent.log_tabular(k, v) for k, v in log_info.items()]
        agent.dump_tabular(print_terminal_info=True)

        if test_agent:
            test_agent.demonstrate()

def main(env_args, agent_args, buffer_args, render=False):
    utils.set_global_seed()

    agent_name = 'Agent'
    agent = Agent(agent_name, agent_args, env_args, device='/gpu:0')

    model = agent_args['model_name']
    print(f'Model {model} starts training')

    test_agent = None
    if render:
        env_args['n_envs'] = 1
        del env_args['max_episode_steps']
        test_agent = Agent(agent_name, agent_args, env_args, save=False, log_tensorboard=False, 
                            log_params=False, log_score=False, device='/gpu:0', reuse=True, graph=agent.graph)

    train(agent, agent_args, test_agent)
