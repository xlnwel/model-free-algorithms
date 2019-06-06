"""
Code for training single agent
"""
import time
import threading
from collections import deque
import numpy as np

from utility import utils
from utility.debug_tools import timeit


def train(agent, render, n_epochs, print_terminal_info=True, background_learning=True):
    interval = 100
    scores_deque = deque(maxlen=interval)
    eps_len_deque = deque(maxlen=interval)
    
    for i in range(1, n_epochs + 1):
        state = agent.env.reset()
        start = time.time()
        action_time = 0

        for _ in range(agent.max_path_length):
            if render:
                agent.env.render()
            t, action = timeit(lambda: agent.act(state))
            action_time += t
            next_state, reward, done, _ = agent.env.step(action)

            agent.add_data(state, action, reward, next_state, done)
            if not background_learning and agent.buffer.good_to_learn:
                agent.learn()
            state = next_state
            if done:
                break

        score = agent.env.get_score()
        eps_len = agent.env.get_length()
        scores_deque.append(score)
        eps_len_deque.append(eps_len)
        avg_score = np.mean(scores_deque)
        avg_eps_len = np.mean(eps_len_deque)
        agent.log_stats(score=score, avg_score=avg_score, eps_len=eps_len, avg_eps_len=avg_eps_len)

        log_info = {
            'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
            'Iteration': i,
            'Time': f'{time.time() - start:3.2f}s',
            'AvgActionTime': action_time / eps_len,
            'Score': score,
            'AvgScore': avg_score,
            'EpsLen': eps_len,
            'AvgEpsLen': avg_eps_len
        }
        [agent.log_tabular(k, v) for k, v in log_info.items()]
        agent.dump_tabular(print_terminal_info=print_terminal_info)

def main(env_args, agent_args, buffer_args, render=False):
    # print terminal information if main is running in the main thread
    utils.set_global_seed()

    algorithm = agent_args['algorithm']
    if algorithm == 'td3':
        from algo.off_policy.td3.agent import Agent
    elif algorithm == 'sac':
        from algo.off_policy.sac.agent import Agent
    else:
        raise NotImplementedError

    agent_args['env_stats']['times'] = 1
    agent = Agent('Agent', agent_args, env_args, buffer_args, log_tensorboard=True, log_score=True, device='/gpu:0')
    if agent_args['background_learning']:
        utils.pwc('Background learning')
        lt = threading.Thread(target=agent.background_learning, daemon=True)
        lt.start()
    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, render, agent_args['n_epochs'], background_learning=agent_args['background_learning'])