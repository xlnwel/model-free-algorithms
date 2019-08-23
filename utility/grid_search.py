import time
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process

from utility.yaml_op import load_args
from utility.debug_tools import assert_colorize


class GridSearch:
    def __init__(self, args_file, train_func, render=False, n_trials=1, dir_prefix=''):
        args = load_args(args_file)
        self.env_args = args['env']
        self.agent_args = args['agent']
        self.buffer_args = args['buffer'] if 'buffer' in args else {}
        self.train_func = train_func
        self.render = render
        self.n_trials = n_trials
        self.dir_prefix = dir_prefix

        self.processes = []

    def __call__(self, **kwargs):
        self._dir_setup()
        if kwargs == {} and self.n_trials == 1:
            # if no argument is passed in, run the default setting
            self.train_func(self.env_args, self.agent_args, self.buffer_args, self.render)        
        else:
            # do grid search
            self.agent_args['model_name'] = 'GS'
            self._change_args(**kwargs)
            [p.join() for p in self.processes]

    def _dir_setup(self):
        # add date to root directory
        now = datetime.now()
        dir_prefix = f'{self.dir_prefix}-' if self.dir_prefix else self.dir_prefix
        dir_fn = lambda filename: (f'logs/'
                                    f'{now.month:02d}{now.day:02d}-'
                                    f'{now.hour:02d}{now.minute:02d}-'
                                    f'{dir_prefix}'
                                    f'{self.agent_args["algorithm"]}-'
                                    f'{self.env_args["name"]}/' 
                                    f'{filename}')
                                    
        dirs = ['model_root_dir', 'log_root_dir']
        for root_dir in dirs:
            self.agent_args[root_dir] = dir_fn(self.agent_args[root_dir])
        if 'video_path' in self.env_args:
            self.env_args['video_path'] = dir_fn(self.env_args['video_path'])

    def _change_args(self, **kwargs):
        if kwargs == {}:
            # basic case
            old_model_name = self.agent_args['model_name']
            for i in range(1, self.n_trials+1):
                if self.n_trials > 1:
                    self.agent_args['model_name'] += f'/trial{i}'
                # arguments should be deep copied here, 
                # otherwise args will be reset if sub-process runs after
                self.env_args['seed'] = 10 * i
                p = Process(target=self.train_func,
                            args=(deepcopy(self.env_args), deepcopy(self.agent_args), 
                                  deepcopy(self.buffer_args), self.render))
                self.agent_args['model_name'] = old_model_name
                p.start()
                time.sleep(1)   # ensure sub-processs start in order
                self.processes.append(p)
        else:
            # recursive case
            kwargs_copy = deepcopy(kwargs)
            key, value = self._popitem(kwargs_copy)

            valid_args = None
            for args in [self.env_args, self.agent_args, self.buffer_args]:
                if key in args:
                    assert_colorize(valid_args is None, f'Conflict: found {key} in both {valid_args} and {args}!')
                    valid_args = args

            err_msg = lambda k, v: f'Invalid Argument: {k}={v}'
            assert_colorize(valid_args is not None, err_msg(key, value))
            if isinstance(value, dict) and len(value) != 0:
                # For simplicity, we do not further consider the case when value is a dict of dicts here
                k, v = self._popitem(value)
                assert_colorize(k in valid_args[key], err_msg(k, v))
                if len(value) != 0:
                    # if there is still something left in value, put value back into kwargs
                    kwargs_copy[key] = value
                self._safe_call(f'-{key}', lambda: self._recursive_trial(valid_args[key], k, v, kwargs_copy))
            else:
                self._recursive_trial(valid_args, key, value, kwargs_copy)

    # helper functions for self._change_args
    def _popitem(self, kwargs):
        assert_colorize(isinstance(kwargs, dict))
        while len(kwargs) != 0:
            k, v = kwargs.popitem()
            if not isinstance(v, list) and not isinstance(v, dict):
                v = [v]
            if len(v) != 0:
                break
        return deepcopy(k), deepcopy(v)

    def _recursive_trial(self, arg, key, value, kwargs):
        assert_colorize(isinstance(value, list), f'Expect value of type list, not {type(value)}: {value}')
        for v in value:
            arg[key] = v
            self._safe_call(f'-{key}={v}', lambda: self._change_args(**kwargs))

    def _safe_call(self, append_name, func):
        old_model_name = self.agent_args['model_name']
        self.agent_args['model_name'] += append_name
        func()
        self.agent_args['model_name'] = old_model_name
