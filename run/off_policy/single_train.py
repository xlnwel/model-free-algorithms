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
    
    for _ in range(1, n_episodes+1):
        state = agent.env.reset()
        score = 0

        for _ in range(1000):
            if render:
                agent.env.render()
            action = agent.act(state)
            next_state, reward, done, _ = agent.env.step(action)
            # if not (np.any(np.isnan(state) | np.isinf(state)) or np.any(np.isnan(next_state) | np.isinf(next_state))):
            agent.add_data(state, action, reward, next_state, done)
            # if agent.buffer.good_to_learn:
            #     agent.learn()
            state = next_state
            score += reward
            if done:
                break

        scores_deque.append(score)
        average_score = np.mean(scores_deque)
        agent.log_score(score, average_score)

        agent.log_tabular('score', score)
        agent.log_tabular('avg_score', average_score)
        agent.dump_tabular()

def main(env_args, agent_args, buffer_args, render=False):
    # print terminal information if main is running in the main thread
    utils.set_global_seed()

    agent_name = 'Agent'
    if 'n_workers' in agent_args:
        del agent_args['n_workers']

    algorithm = agent_args['algorithm']
    if algorithm == 'td3':
        from td3.agent import Agent
    elif algorithm == 'sac':
        from sac.agent import Agent
    else:
        raise NotImplementedError

    agent = Agent(agent_name, agent_args, env_args, buffer_args, log_tensorboard=True, log_score=True, device='/gpu:0')
    lt = threading.Thread(target=agent.background_learning, daemon=True)
    lt.start()
    model = Path(agent_args['model_dir']) / agent_args['model_name']
    print(f'Model {model} starts training')
    
    train(agent, render)