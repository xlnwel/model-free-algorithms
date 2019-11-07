import os, random
import os.path as osp
import argparse
import math
import multiprocessing
import numpy as np
import sympy

from utility.display import assert_colorize


def to_int(s):
    return int(float(s))
    
def isscalar(x):
    return isinstance(x, (int, float))
    
def moments(x, mask=None):
    if mask is None:
        x_mean = np.mean(x)
        x_std = np.std(x)
    else:
        # expand mask to match the dimensionality of x
        while len(mask.shape) < len(x.shape):
            mask = mask[..., None]
        # compute valid entries in x corresponding to True in mask
        n = np.sum(mask)
        for i in range(len(mask.shape)):
            if mask.shape[i] != 1:
                assert_colorize(mask.shape[i] == x.shape[i], 
                        f'{i}th dimension of mask{mask.shape[i]} does not match'
                        f'that of x{x.shape[i]}')
            else:
                n *= x.shape[i]
        # compute x_mean and x_std from entries in x corresponding to True in mask
        x_mask = x * mask
        x_mean = np.sum(x_mask) / n
        x_std = np.sqrt(np.sum(mask * (x_mask - x_mean)**2) / n)
    
    return x_mean, x_std
    
def standardize(x, epsilon=1e-8, mask=None):
    if mask is not None:
        while len(mask.shape) < len(x.shape):
            mask = mask[..., None]
    x_mean, x_std = moments(x, mask)
    x = (x - x_mean) / (x_std + epsilon)
    if mask is not None:
        x *= mask
    return x

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_main_process():
    return multiprocessing.current_process().name == 'MainProcess'

def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_available_gpus():
    # recipe from here:
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
 
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def timeformat(t):
    return f'{t:.2e}'

def squarest_grid_size(num_images, more_on_width=True):
    """Calculates the size of the most square grid for num_images.

    Calculates the largest integer divisor of num_images less than or equal to
    sqrt(num_images) and returns that as the width. The height is
    num_images / width.

    Args:
        num_images: The total number of images.
        more_on_width: If cannot fit in a square, put more cells on width
    Returns:
        A tuple of (height, width) for the image grid.
    """
    divisors = sympy.divisors(num_images)
    square_root = math.sqrt(num_images)
    for d in divisors:
        if d > square_root:
            break
    h, w = (num_images // d, d) if more_on_width else (d, num_images // d)

    return (h, w)

def check_make_dir(path):
    _, ext = osp.splitext(path)
    if ext: # if path is a file path, extract its directory path
        path, _ = osp.split(path)

    if not os.path.isdir(path):
        os.mkdir(path)

def zip_pad(*args):
    list_len = None
    for x in args:
        if isinstance(x, list) or isinstance(x, tuple):
            list_len = len(x)
            break
    assert list_len is not None
    new_args = []
    for i, x in enumerate(args):
        if not isinstance(x, list) and not isinstance(x, tuple):
            new_args.append([x] * list_len)
        else:
            new_args.append(x)

    return list(zip(*new_args))
    
def convert_indices(indices, *args):
    """ 
    convert 1d indices to a tuple of for ndarray index
    args specify the size of the first len(args) dimensions
    e.g.
    x = np.array([[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]])
    print(x.shape)
    >>> (2, 2, 2)
    indices = np.random.randint(7, size=5)
    print(indices)
    >>> [6 6 0 3 1]
    indices = convert_shape(indices, *x.shape)
    print(indices)
    >>> (array([1, 1, 0, 0, 0]), array([1, 1, 0, 1, 0]), array([0, 0, 0, 1, 1]))
    print(x[indices])
    >>> array(['b0', 'c1', 'b1', 'a1', 'c0'])
    """
    res = []
    v = indices
    for i in range(1, len(args)):
        prod = np.prod(args[i:])
        res.append(v // prod)
        v = v % prod
    res.append(v)

    return tuple(res)
