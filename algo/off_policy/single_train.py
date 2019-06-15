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
    
    acttimes = deque(maxlen=1000)
    learntimes = deque(maxlen=1000)
    itrtimes = deque(maxlen=1000)
    for i in range(1, n_epochs + 1):
        state = agent.env.reset()
        start = time.time()

        for _ in range(agent.max_path_length):
            if render:
                agent.env.render()
            at, action = timeit(lambda: agent.act(state) if agent.buffer.good_to_learn else agent.env.random_action())
            acttimes.append(at)
            next_state, reward, done, _ = agent.env.step(action)

            agent.add_data(state, action, reward, done)
            if not background_learning and agent.buffer.good_to_learn:
                lt, _ = timeit(lambda: agent.learn())
                learntimes.append(lt)
            state = next_state
            if done:
                break

        # bookkeeping
        score = agent.env.get_score()
        eps_len = agent.env.get_length()
        itr_time = (time.time() - start) / eps_len
        itrtimes.append(itr_time)
        scores_deque.append(score)
        eps_len_deque.append(eps_len)
        avg_score = np.mean(scores_deque)
        avg_eps_len = np.mean(eps_len_deque)
        if hasattr(agent, 'stats'):
            agent.record_stats(score=score, avg_score=avg_score, eps_len=eps_len, avg_eps_len=avg_eps_len)

        log_info = {
            'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
            'Iteration': i,
            'IterationTime': utils.timeformat(np.mean(itrtimes)) + 's',
            'ActionTime': utils.timeformat(np.mean(acttimes)) + 's',
            'LearnTime': (utils.timeformat(np.mean(learntimes)) if learntimes else '0') + 's',
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
    elif algorithm == 'rainbow-iqn':
        from algo.off_policy.rainbow_iqn.agent import Agent
    else:
        raise NotImplementedError

    agent_args['env_stats']['times'] = 1
    agent = Agent('Agent', agent_args, env_args, buffer_args, log_tensorboard=False, log_stats=True, save=False, device='/GPU:0')
    if agent_args['background_learning']:
        utils.pwc('Background Learning...')
        lt = threading.Thread(target=agent.background_learning, daemon=True)
        lt.start()
    else:
        utils.pwc('Foreground Learning...')
    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, render, agent_args['n_epochs'], background_learning=agent_args['background_learning'])