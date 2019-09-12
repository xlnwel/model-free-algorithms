"""
Code for training single agent
"""
import time
import threading
from collections import deque
import numpy as np
import tensorflow as tf

from utility import utils
from utility.debug_tools import timeit


def run_trajectory(agent, fn, render):
    """ run a trajectory, fn is a function executed after each environment step """
    env = agent.env
    state = env.reset()
    for i in range(env.max_episode_steps):
        if render:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        fn(state, action, reward, done, i)
        state = next_state
        if done:
            break

    return env.get_score(), env.get_length()

def train(agent, render, n_epochs, print_terminal_info=True, background_learning=True):
    def train_fn(state, action, reward, done, i):
        agent.add_data(state, action, reward, done)
        if not background_learning and agent.buffer.good_to_learn and i % agent.args['update_freq'] == 0:
            agent.learn()
    def eval_fn(state, action, reward, done, i):
        pass    # do nothing at eval time
    
    interval = 100
    scores = deque(maxlen=interval)
    eps_lens = deque(maxlen=interval)
    
    for k in range(1, n_epochs + 1):
        score, eps_len = run_trajectory(agent, train_fn, False)
        print(f'\rIteration {k}, Score={score}', end='')

        # evaluate every 100 episodes
        if k % 100 == 0:
            for i in range(1, interval+1):
                start = time.time()
                score, eps_len = run_trajectory(agent, eval_fn, render)
                scores.append(score)
                eps_lens.append(eps_len)

                if i % 10 == 0:
                    steptime = (time.time() - start) / eps_len

                    score_mean = np.mean(scores)
                    score_std = np.std(scores)
                    eps_len_mean = np.mean(eps_lens)
                    eps_len_std = np.std(eps_lens)
                    if hasattr(agent, 'stats'):
                        agent.record_stats(score_mean=score_mean, score_std=score_std,
                                            eps_len_mean=eps_len_mean, eps_len_std=eps_len_std)
                    
                    log_info = {
                        'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
                        'Iteration': k-100 + i,
                        'StepTime': utils.timeformat(np.mean(steptime)) + 's',
                        'ScoreMean': score_mean,
                        'ScoreStd': score_std,
                        'EpsLenMean': eps_len_mean,
                        'EpsLenStd': eps_len_std
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
    agent = Agent('Agent', agent_args, env_args, buffer_args, 
                    log_tensorboard=False, log_stats=True, save=False, 
                    device='/GPU: 0')
    if agent_args['background_learning']:
        utils.pwc('Background Learning...')
        lt = threading.Thread(target=agent.background_learning, daemon=True)
        lt.start()
    else:
        utils.pwc('Foreground Learning...')
    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, render, agent_args['n_epochs'], background_learning=agent_args['background_learning'])