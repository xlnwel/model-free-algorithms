import numpy as np


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
    if isinstance(string, list) or isinstance(string, tuple):
        for s in string:
            print(colorize(s, color, bold, highlight))
    else:
        print(colorize(string, color, bold, highlight))

def assert_colorize(cond, err_msg=''):
    assert cond, colorize(err_msg, 'red')

def display_var_info(vars, name='trainable'):
    pwc(f'Print {name} variables', 'yellow')
    count_params = 0
    for v in vars:
        name = v.name
        if '/Adam' in name or 'beta1_power' in name or 'beta2_power' in name: continue
        v_params = int(np.prod(v.shape.as_list()))
        count_params += v_params
        if '/b:' in name or '/biases' in name: continue    # Wx+b, bias is not interesting to look at => count params, but not print
        pwc(f'   {name}{" "*(100-len(name))} {v_params:d} params {v.shape}', 'yellow')

    pwc(f'Total model parameters: {count_params*1e-6:0.2f} million', 'yellow')
	