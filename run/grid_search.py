from copy import deepcopy
from multiprocessing import Process

from utility import yaml_op, debug_tools


class GridSearch:
    def __init__(self, args_file, train_func, render=False, n_trials=1):
        args = yaml_op.load_args(args_file)
        self.env_args = args['env']
        self.agent_args = args['agent']
        self.buffer_args = args['buffer']
        self.train_func = train_func
        self.render = render
        self.n_trials = n_trials

        if 'num_workers' in self.agent_args:
            del self.agent_args['num_workers']

        self.agent_args['model_dir'] = f"{self.agent_args['algorithm']}-{self.agent_args['model_dir']}"

        self.processes = []

    def __call__(self, **kwargs):
        if kwargs == {} and self.n_trials == 1:
            # if no argument is passed in, run the default setting
            self.train_func(self.env_args, self.agent_args, self.buffer_args, self.render)
            
        # do grid search
        from datetime import datetime
        now = datetime.now()
        self.agent_args['model_name'] = f'{now.month:02d}{now.day:02d}'
        self._change_args(**kwargs)
        [p.join() for p in self.processes]

    def _change_args(self, **kwargs):
        if kwargs == {}:
            old_model_name = self.agent_args['model_name']
            for i in range(self.n_trials):
                self.agent_args['model_name'] += f'-trial{i}'
                # arguments should be deep copied here, otherwise args will be reset
                p = Process(target=self.train_func,
                            args=(deepcopy(self.env_args), deepcopy(self.agent_args), 
                                  deepcopy(self.buffer_args), self.render))
                self.agent_args['model_name'] = old_model_name
                p.start()
                self.processes.append(p)
        else:
            key, value = kwargs.popitem()
            valid_args = []
            for args in [self.agent_args, self.buffer_args, self.env_args]:
                valid_arg = args if key in args else False
                if valid_arg != False:
                    valid_args.append(valid_arg)
                    break
            
            err_msg = lambda k, v: debug_tools.colorize(f'Invalid Argument: {k}={v}', 'red')
            assert valid_arg != False, err_msg(key, value)

            # here we don't consider the case when value is a dict of dicts
            if isinstance(value, dict):
                for k, v in value.items():
                    assert k in valid_arg[key], err_msg(k, v)

            old_model_name = self.agent_args['model_name']

            def recursive_trial(arg, key, value, kwargs):
                assert isinstance(value, list), err_msg(key, value)
                for v in value:
                    arg[key] = v
                    self.agent_args['model_name'] += f'-{key}={v}'
                    self._change_args(**kwargs)
                    self.agent_args['model_name'] = old_model_name
            if isinstance(value, dict):
                for k, v in value.items():
                    recursive_trial(valid_arg[key], k, v, kwargs)
            else:
                recursive_trial(valid_arg, key, value, kwargs)

    def _find(self, args, key):
        if key in args:
            return args

        return False
