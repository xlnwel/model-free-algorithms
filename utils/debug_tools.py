from time import strftime, gmtime, time

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
	