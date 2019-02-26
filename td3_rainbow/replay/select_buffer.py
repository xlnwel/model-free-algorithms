from td3_rainbow.replay.uniform_replay import UniformReplay
from td3_rainbow.replay.rank_based_replay import RankBasedPrioritizedReplay
from td3_rainbow.replay.proportional_replay import ProportionalPrioritizedReplay


def construct_buffer(buffer_args):
    buffer_type = buffer_args['type']
    batch_size = buffer_args['batch_size']
    n_steps = buffer_args['n_steps']
    gamma = buffer_args['gamma']

    if buffer_type == 'uniform':
        buffer = UniformReplay(buffer_args, batch_size, n_steps=n_steps, gamma=gamma)
    elif buffer_type == 'rank':
        buffer = RankBasedPrioritizedReplay(buffer_args, batch_size, n_steps=n_steps, gamma=gamma)
    elif buffer_type == 'proportional':
        buffer = ProportionalPrioritizedReplay(buffer_args, batch_size, n_steps=n_steps, gamma=gamma)
    else:
        raise ValueError('Invalid buffer type.')

    return buffer