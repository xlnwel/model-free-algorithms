import os,sys
import argparse
import logging

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
# print(sys.path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(os.environ['PYTHONPATH'])

from run.grid_search import GridSearch

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm',
                        type=str,
                        choices=['td3', 'sac', 'ppo'])
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

    distributed = True if cmd_args.distributed == 'true' else False
    if algorithm == 'td3':
        arg_file = 'td3/args.yaml'
        if distributed:
            from run.off_policy.distributed_train import main
        else:
            from run.off_policy.single_train import main
    elif algorithm == 'sac':
        arg_file = 'sac/args.yaml'
        if distributed:
            from run.off_policy.distributed_train import main
        else:
            from run.off_policy.single_train import main
    elif algorithm == 'ppo':
        arg_file = 'ppo/args.yaml'
        from run.on_policy.train import main
    else:
        raise NotImplementedError

    # disable tensorflow debug information
    # logging.getLogger("tensorflow").setLevel(logging.WARNING)

    render = True if cmd_args.render == 'true' else False

    gs = GridSearch(arg_file, main, render, n_trials=cmd_args.trials)

    # Grid search happens here
    # gs(ac={'shared_fc_units': 0, 'lstm_units': 0, 'actor_units': [[64, 64]], 'critic_units': [[64, 64]]})
    gs(ac={'shared_fc_units': 0, 'lstm_units': 0})