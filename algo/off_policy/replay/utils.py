import numpy as np

from utility.utils import assert_colorize


def init_buffer(buffer, capacity, state_space, action_dim, has_priority):
    target_buffer = {'priority': np.zeros((capacity, 1))} if has_priority else {}
    target_buffer.update({
        'state': np.zeros((capacity, *state_space)),
        'action': np.zeros((capacity, action_dim)),
        'reward': np.zeros((capacity, 1)),
        'next_state': np.zeros((capacity, *state_space)),
        'done': np.zeros((capacity, 1)),
        'steps': np.zeros((capacity, 1))
    })

    buffer.update(target_buffer)

def reset_buffer(buffer):
    target_buffer = {}
    for k, v in buffer.items():
        target_buffer[k] = np.zeros_like(v)

    buffer.update(target_buffer)

def add_buffer(buffer, idx, state, action, reward, next_state, done, n_steps, gamma):
    buffer['state'][idx] = state
    buffer['action'][idx] = action
    buffer['reward'][idx] = reward
    buffer['next_state'][idx] = next_state
    buffer['done'][idx] = done
    buffer['steps'][idx] = 1
    # Update previous experience if multi-step is required
    for i in range(1, n_steps):
        k = idx - i
        if buffer['done'][k] == True:
            # Do not continue updating when done is encountered
            break
        buffer['reward'][k] += gamma**i * reward
        buffer['next_state'][k] = next_state
        buffer['done'][k] = done
        buffer['steps'][k] += 1

def copy_buffer(dest_buffer, dest_start, dest_end, orig_buffer, orig_start, orig_end):
    assert_colorize(dest_end - dest_start == orig_end - orig_start, 'Inconsistent lengths of dest_buffer and orig_buffer.')
    if dest_end - dest_start == 0:
        return
    
    for key in dest_buffer.keys():
        dest_buffer[key][dest_start: dest_end] = orig_buffer[key][orig_start: orig_end]
