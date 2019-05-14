from collections import namedtuple
from queue import Queue
import numpy as np

from utility.utils import normalize, assert_colorize
from algo.on_policy.ppo.agent import Agent
from algo.on_policy.impala.buffer import IMPALABuffer


class Learner(Agent):
    def __init__(self,
                 name,
                 args,
                 env_args,
                 sess_config=None,
                 save=True,
                 log_tensorboard=True,
                 log_params=False,
                 log_score=True,
                 device=None,
                 reuse=None,
                 graph=None):
        
        super().__init__(name, args, 
                         env_args,
                         sess_config=sess_config,
                         save=save, 
                         log_tensorboard=log_tensorboard,
                         log_params=log_params, 
                         log_score=log_score,
                         device=device,
                         reuse=reuse,
                         graph=graph)

        self.buffer = IMPALABuffer(args['max_buffer_size'], args['seq_len'], 
                                   self.env_vec.state_space, 
                                   self.env_vec.action_dim)
        
        