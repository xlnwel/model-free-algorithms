from time import strftime, gmtime, time

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

def timeit(func, name=None, to_print=False):
	start_time = gmtime()
	start = time()
	result = func()
	end = time()
	end_time = gmtime()

	if to_print:
		print('{}: Start "{}", End "{}", Duration "{:.2f}s"'.format(name if name else func.__name__, 
															 strftime("%d %b %H:%M:%S", start_time), 
													 	 	 strftime("%d %b %H:%M:%S", end_time), 
													 		 end - start))

	return end - start, result
	