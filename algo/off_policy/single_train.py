"""
Code for training single agent
"""

import threading
from pathlib import Path
from collections import deque
import numpy as np

from utility import utils


def train(agent, render, n_episodes=3000, print_terminal_info=False):
    interval = 100
    scores_deque = deque(maxlen=interval)
    eps_len_deque = deque(maxlen=interval)
    
    for _ in range(1, n_episodes+1):
        state = agent.env.reset()

        for _ in range(1000):
            if render:
                agent.env.render()
            action = agent.act(state)
            next_state, reward, done, _ = agent.env.step(action)

            agent.add_data(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        score = agent.env.get_episode_score()
        eps_len = agent.env.get_episode_length()
        scores_deque.append(score)
        eps_len_deque.append(eps_len)
        avg_score = np.mean(scores_deque)
        avg_eps_len = np.mean(eps_len_deque)
        agent.log_stats(score=score, avg_score=avg_score, eps_len=eps_len, avg_eps_len=avg_eps_len)

        agent.log_tabular('score', score)
        agent.log_tabular('avg_score', avg_score)
        agent.log_tabular('eps_len', eps_len)
        agent.log_tabular('avg_eps_len', avg_eps_len)
        agent.dump_tabular()

def main(env_args, agent_args, buffer_args, render=False):
    # print terminal information if main is running in the main thread
    utils.set_global_seed()

    agent_name = 'Agent'
    if 'n_workers' in agent_args:
        del agent_args['n_workers']

    algorithm = agent_args['algorithm']
    if algorithm == 'td3':
        from algo.off_policy.td3.agent import Agent
    elif algorithm == 'sac':
        from algo.off_policy.sac.agent import Agent
    else:
        raise NotImplementedError

    agent_args['env_stats']['times'] = 1
    agent = Agent(agent_name, agent_args, env_args, buffer_args, log_tensorboard=True, log_score=True, device='/gpu:0')
    lt = threading.Thread(target=agent.background_learning, daemon=True)
    lt.start()
    model = Path(agent_args['model_dir']) / agent_args['model_name']
    print(f'Model {model} starts training')
    
    train(agent, render)