import argparse
import logging

from run.grid_search import GridSearch

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm',
                        type=str,
                        choices=['td3', 'sac'])
    parser.add_argument('--render',
                        type=str,
                        choices=['true', 'false'],
                        default='false')
    parser.add_argument('--distributed',
                        type=str,
                        choices=['true', 'false'],
                        default='false')
    parser.add_argument('--trials',
                        type=int,
                        default=1)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = cmd_args.algorithm

    if algorithm == 'td3':
        arg_file = 'td3/args.yaml'
    elif algorithm == 'sac':
        arg_file = 'sac/args.yaml'
    else:
        raise NotImplementedError

    # disable tensorflow debug information
    # logging.getLogger("tensorflow").setLevel(logging.WARNING)

    render = True if cmd_args.render == 'true' else False
    distributed = True if cmd_args.distributed == 'true' else False
    if distributed:
        from run.off_policy.distributed_train import main
    else:
        from run.off_policy.single_train import main

    gs = GridSearch(arg_file, main, render, n_trials=cmd_args.trials)

    # Grid search happens here
    gs(num_workers=[10], actor={'units':[[500, 300]]})
