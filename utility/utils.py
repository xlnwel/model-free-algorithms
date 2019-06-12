import os, random
import multiprocessing
import numpy as np
import tensorflow as tf


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return f'\x1b[{";".join(attr)}m{string}\x1b[0m'

def pwc(string, color='red', bold=False, highlight=False):
    """
    Print with color
    """
    print(colorize(string, color, bold, highlight))

def normalize(x, mean=0., std=1., epsilon=1e-8):
    x = (x - np.mean(x)) / (np.std(x) + epsilon)
    x = x * std + mean

    return x

def schedule(start_value, step, decay_steps, decay_rate):
    return start_value * decay_rate**(step // decay_steps)

def is_main_process():
    return multiprocessing.current_process().name == 'MainProcess'

def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_available_gpus():
    # recipe from here:
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
 
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def timeformat(t):
    return f'{t:.2e}'
