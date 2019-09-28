from time import strftime, gmtime, time
import numpy as np

from utility.utils import colorize, pwc


def timeit(func, args=[], name=None, to_print=False):
	start_time = gmtime()
	start = time()
	result = func(*args)
	end = time()
	end_time = gmtime()

	if to_print:
		print('{}: Start "{}", End "{}", Duration "{:.2f}s"'.format(name if name else func.__name__, 
                                                                    strftime("%d %b %H:%M:%S", start_time), 
                                                                    strftime("%d %b %H:%M:%S", end_time), 
                                                                    end - start))

	return end - start, result

def assert_colorize(cond, err_msg=''):
    assert cond, colorize(err_msg, 'red')

def display_var_info(vars, name='trainable'):
    pwc(f'Print {name} variables', 'yellow')
    count_params = 0
    for v in vars:
        name = v.name
        if '/Adam' in name or 'beta1_power' in name or 'beta2_power' in name: continue
        v_params = np.prod(v.shape.as_list())
        count_params += v_params
        if '/b:' in name or '/biases' in name: continue    # Wx+b, bias is not interesting to look at => count params, but not print
        pwc(f'   {name}{" "*(100-len(name))} {v_params:d} params {v.shape}', 'yellow')

    pwc(f'Total model parameters: {count_params*1e-6:0.2f} million', 'yellow')
	