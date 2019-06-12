"""
Some simple logging functionality, inspired by rllab's logging

The original code is from homework of Berkeley cs294-112: 
http://rail.eecs.berkeley.edu/deeprlcourse/

Adapted by S.C.

"""
import os.path as osp
import os, time, atexit

from utility.utils import pwc
from utility.debug_tools import assert_colorize


class Logger:
    def __init__(self, log_dir, log_file='log.txt', exp_name=None):
        """
        Initialize a Logger.

        Args:
            log_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            log_file (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        log_file = log_file if log_file.endswith('.txt') else log_file + '.txt'
        self.log_dir = log_dir or f"/tmp/experiments/{time.time()}"
        if osp.exists(self.log_dir):
            print(f"Warning: Log dir {self.log_dir} already exists! Storing info there anyway.")
        else:
            os.makedirs(self.log_dir)
        self.output_file = open(osp.join(self.log_dir, log_file), 'w')
        atexit.register(self.output_file.close)
        pwc("Logging data to %s"%self.output_file.name, 'green', bold=True)

        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        pwc(msg, color, bold=True)

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert_colorize(key in self.log_headers, f"Trying to introduce a new key {key} that you didn't include in the first iteration")
        assert_colorize(key not in self.log_current_row, f"You already set {key} this iteration. Maybe you forgot to call dump_tabular()")
        self.log_current_row[key] = val
    
    def dump_tabular(self, print_terminal_info=False):
        """
        Write all of the diagnostics from the current iteration.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        n_slashes = 22 + max_key_len
        if print_terminal_info:
            print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else val
            if print_terminal_info:
                print(f'| {key:>{max_key_len}s} | {valstr:>15s} |')
            vals.append(val)
        if print_terminal_info:
            print("-"*n_slashes)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False
