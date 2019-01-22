import yaml
import numpy as np
from pathlib import Path

def norm(x, mean=0., std=1., epsilon=1e-8):
    normalized_x = (x - np.mean(x)) / (np.std(x) + epsilon)
    x = normalized_x * std + mean

    return x

def default_path(filename):
    if filename.startswith('/'):
        return Path(filename)
    else:
        return Path('.') / filename

# load arguments from args.yaml
def load_args(filename='args.yaml'):
    with open(default_path(filename), 'r') as f:
        try:
            yaml_f = yaml.load(f)
            return yaml_f
        except yaml.YAMLError as exc:
            print(exc)

# save args to args.yaml
def save_args(args, args_to_update={}, filename='args.yaml'):
    assert isinstance(args, dict)
    
    filepath = default_path(filename)
    if filepath.exists():
        if args_to_update is None:
            args_to_update = load_args(filename)
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

    with filepath.open('w') as f:
        try:
            args_to_update.update(args)
            yaml.dump(args_to_update, f)
        except yaml.YAMLError as exc:
            print(exc)
    