import os, sys
import argparse
import logging
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.utils import str2bool
from utility.yaml_op import load_args
from utility.debug_tools import assert_colorize


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        choices=['td3', 'sac', 'apex-td3', 'apex-sac', 'ppo', 'a2c',
                                 'rainbow-iqn'])
    parser.add_argument('--background', '-b',
                        type=str2bool,
                        choices=[True, False],
                        default=False)
    parser.add_argument('--render', '-r',
                        action='store_true')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--file', '-f',
                        type=str,
                        default='',
                        help='filepath to restore')
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if algorithm == 'td3' or algorithm == 'sac':
        from algo.off_policy.single_train import main
    elif algorithm == 'rainbow-iqn':
        from algo.off_policy.single_train import main
    elif algorithm.startswith('apex'):
        from algo.off_policy.distributed_train import main
    elif algorithm == 'ppo':
        from algo.on_policy.single_train import main
    elif algorithm == 'a2c':
        from algo.on_policy.distributed_train import main
    else:
        raise NotImplementedError

    return main
    
def get_arg_file(algorithm):
    if algorithm == 'td3':
        arg_file = 'algo/off_policy/td3/args.yaml'
    elif algorithm == 'sac':
        arg_file = 'algo/off_policy/sac/args.yaml'
    elif algorithm == 'rainbow-iqn':
        arg_file = 'algo/off_policy/rainbow_iqn/args.yaml'
    elif algorithm == 'apex-td3':
        arg_file = 'algo/off_policy/apex/td3_args.yaml'
    elif algorithm == 'apex-sac':
        arg_file = 'algo/off_policy/apex/sac_args.yaml'
    elif algorithm == 'ppo':
        arg_file = 'algo/on_policy/ppo/args.yaml'
    elif algorithm == 'a2c':
        arg_file = 'algo/on_policy/a2c/args.yaml'
    else:
        raise NotImplementedError

    return arg_file

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = cmd_args.algorithm
    
    arg_file = get_arg_file(algorithm)
    main = import_main(algorithm)
    
    render = cmd_args.render
    background_learning = cmd_args.background

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
        if cmd_args.background != '':
            background_learning = True if cmd_args.background == 'true' else False

        main(env_args, agent_args, buffer_args, render=render)
    else:
        prefix = cmd_args.prefix
        if algorithm.startswith('apex'):
            prefix = f'{prefix}-apex' if prefix else 'apex'
        # Although random parameter search is in general better than grid search, 
        # we here continue to go with grid search since it is easier to deal with architecture search
        gs = GridSearch(arg_file, main, render=render, n_trials=cmd_args.trials, dir_prefix=prefix)
        if background_learning:
            gs.agent_args['background_learning'] = background_learning

        # Grid search happens here
        if algorithm == 'ppo':
            gs(ac=dict(policy_lr=[3e-4], value_lr=[6e-4, 1e-3]))
        elif algorithm == 'a2c':
            gs()
        elif algorithm == 'td3':
            gs(update_freq=[1, 10])
        elif algorithm == 'sac':
            gs()
        elif algorithm == 'rainbow-iqn':
            gs()
        elif algorithm == 'apex-td3':
            gs()
        elif algorithm == 'apex-sac':
            gs()
        else:
            raise NotImplementedError
