import os, sys
import argparse
import logging
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.yaml_op import load_args
from utility.utils import assert_colorize


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
    parser.add_argument('--file', '-f',
                        type=str,
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

    if cmd_args.file != '':
        args = load_args(arg_file)
        env_args = args['env']
        agent_args = args['agent']
        buffer_args = args['buffer'] if 'buffer' in args else {}
        model_file = cmd_args.file
        assert_colorize(os.path.exists(model_file), 'Model file does not exists')
        agent_args['model_root_dir'], agent_args['model_name'] = os.path.split(model_file)
        agent_args['log_root_dir'], _ = os.path.split(agent_args['model_root_dir'])
        agent_args['log_root_dir'] += '/logs'

        main(env_args, agent_args, buffer_args, render=render)
    else:
        prefix = cmd_args.prefix + ('dist' if distributed else '') 
        gs = GridSearch(arg_file, main, render, n_trials=cmd_args.trials, dir_prefix=prefix)

        args = {'units': [(1024, 512, 256), (1024, 512, 512, 256)]}
        # Grid search happens here
        if algorithm == 'ppo':
            gs(n_envs=[10, 20])
        elif algorithm == 'td3':
            gs()
        elif algorithm == 'sac':
            gs()
