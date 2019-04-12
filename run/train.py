import os,sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        choices=['td3', 'sac', 'ppo'])
    parser.add_argument('--render', '-r',
                        type=str,
                        choices=['true', 'false'],
                        default='false')
    parser.add_argument('--distributed', '-d',
                        type=str,
                        choices=['true', 'false'],
                        default='false')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1)
    parser.add_argument('--prefix', '-p',
                        default='')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = cmd_args.algorithm

    distributed = True if cmd_args.distributed == 'true' else False
    if distributed:
        if algorithm == 'td3' or algorithm == 'sac':
            from algo.off_policy.distributed_train import main
        elif algorithm == 'ppo':
            from algo.on_policy.distributed_train import main
    else:
        if algorithm == 'td3' or algorithm == 'sac':
            from algo.off_policy.single_train import main
        elif algorithm == 'ppo':
            from algo.on_policy.single_train import main
    
    if algorithm == 'td3':
        arg_file = 'algo/off_policy/td3/args.yaml'
    elif algorithm == 'sac':
        arg_file = 'algo/off_policy/sac/args.yaml'
    elif algorithm == 'ppo':
        arg_file = 'algo/on_policy/ppo/args.yaml'
    else:
        raise NotImplementedError

    render = True if cmd_args.render == 'true' else False

    gs = GridSearch(arg_file, main, render, n_trials=cmd_args.trials, dir_prefix=cmd_args.prefix)

    # Grid search happens here
    if algorithm == 'ppo':
        gs(ac={'actor_units': (512, 256, 256), 'critic_units': (512, 512, 256)})
    elif algorithm == 'td3':
        gs()
        # gs(actor={'units': (64, 64)}, critic={'units': (64, 64)}, name='LunarLanderContinuous-v2', n_epochs=5000)
    elif algorithm == 'sac':
        gs(temperature=[.2, .01])
