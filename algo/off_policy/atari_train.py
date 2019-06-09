"""
Code for training single agent in atari
"""
import time
import threading
from collections import deque
import numpy as np

from utility import utils
from utility.debug_tools import timeit


def train(agent, render, log_steps, print_terminal_info=True, background_learning=True):
    state = agent.env.reset()
    t = 0
    acttimes = deque(maxlen=log_steps)
    envtimes = deque(maxlen=log_steps)
    addtimes = deque(maxlen=log_steps)
    learntimes = deque(maxlen=log_steps)
    episode_lengths = deque(maxlen = 100)
    el = 0
    while agent.env.get_total_steps() < 2e8:
        el += 1
        t += 1
        if render:
            agent.env.render()
        if not agent.buffer.good_to_learn:
            acttime, action = timeit(lambda: agent.env.random_action())
        else:
            acttime, action = timeit(lambda:agent.atari_act(state))
        acttimes.append(acttime)

        envtime, (next_state, reward, done, _) = timeit(lambda:agent.env.step(action))
        envtimes.append(envtime)
        
        addtime, _ = timeit(lambda:agent.add_data(state, action, reward, next_state, done))
        addtimes.append(addtime)
        if not background_learning and agent.buffer.good_to_learn:
            learntime, _ = timeit(lambda:agent.atari_learn(t))
            learntimes.append(learntime)

        state = agent.env.reset() if done else next_state
        if done:
            episode_lengths.append(el)
            el = 0

        if t % log_steps == 0:
            episode_scores = agent.env.get_episode_scores()
            # episode_lengths = agent.env.get_episode_lengths()
            eps_len = agent.env.get_length()
            score = episode_scores[-1]
            avg_score = np.mean(episode_scores[-100:])
            eps_len = episode_lengths[-1]
            avg_eps_len = np.mean(episode_lengths)
            agent.log_stats(score=score, avg_score=avg_score, eps_len=eps_len, avg_eps_len=avg_eps_len)

            log_info = {
                'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
                'Timestep': f'{(t//1000):3d}k',
                'ActTime': utils.timeformat(np.mean(acttimes)),
                'EnvTime': utils.timeformat(np.mean(envtimes)),
                'AddTime': utils.timeformat(np.mean(addtimes)),
                'LearnTime': utils.timeformat(np.mean(learntimes) if learntimes else 0),
                'Iteration': len(episode_scores),
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
    if algorithm == 'rainbow':
        from algo.off_policy.rainbow.agent import Agent
    else:
        raise NotImplementedError

    agent_args['env_stats']['times'] = 1
    agent = Agent('Agent', agent_args, env_args, buffer_args, log_tensorboard=True, log_score=True, log_params=True, device='/gpu:0')
    if agent_args['background_learning']:
        utils.pwc('Background Learning...')
        lt = threading.Thread(target=agent.background_learning, daemon=True)
        lt.start()
    else:
        utils.pwc('Foreground Learning...')
    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, render, log_steps=int(1e4), background_learning=agent_args['background_learning'])