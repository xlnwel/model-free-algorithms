import os
import time
import numpy as np

from utility import utils
from algo.on_policy.ppo.agent import Agent
import ray

def train(agent, agent_args, test_agent):
    for i in range(1, agent_args['n_epochs'] + 1):
        start = time.time()
        env_stats = agent.sample_trajectories()

        loss_info_list = agent.optimize(i)

        loss_info = list(zip(*loss_info_list))
        ppo_loss, entropy, approx_kl, p_clip_frac, v_clip_frac, value_loss = loss_info

        # logging
        scores, eps_lens = env_stats
        agent.store(
            score_mean=np.mean(scores),
            score_std=np.std(scores),
            score_max=np.max(scores),
            score_min=np.min(scores),
            epslen_mean=np.mean(eps_lens),
            ppo_loss=np.mean(ppo_loss),
            value_loss=np.mean(value_loss),
            entropy=np.mean(entropy),
            approx_kl=np.mean(approx_kl),
            p_clip_frac=np.mean(p_clip_frac),
            V_clip_frac=np.mean(v_clip_frac)
        )
        
        agent.record_stats(agent.get_stored_stats())
        agent.log_stats(i, 'Train')

        if test_agent:
            test_agent.demonstrate()

def main(env_args, agent_args, buffer_args, render=False):
    utils.set_global_seed()

    if env_args.get('n_workers', 0) > 1:
        ray.init()
    agent_name = 'Agent'
    agent = Agent(agent_name, agent_args, env_args, 
                  save=False, log=True, log_tensorboard=True, 
                  log_params=False, log_stats=True, device='/gpu:0')

    test_agent = None
    if render:
        env_args['n_envs'] = 1  # run test agent in a single environment
        env_args['log_video'] = True
        test_agent = Agent(agent_name, agent_args, env_args, save=False, log_tensorboard=False, 
                            log_params=False, log_stats=False, device='/gpu:0', reuse=True, graph=agent.graph)

    train(agent, agent_args, test_agent)
