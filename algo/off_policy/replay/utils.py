import numpy as np

from utility.debug_tools import assert_colorize


def init_buffer(buffer, capacity, state_space, action_dim, has_priority):
    state_dtype = np.float16
    action_shape = (capacity, ) if action_dim == 1 else (capacity, action_dim)
    action_dtype = np.int8 if action_dim == 1 else np.float16

    target_buffer = {'priority': np.zeros((capacity, 1))} if has_priority else {}
    target_buffer.update({
        'state': np.zeros((capacity, *state_space), dtype=state_dtype),
        'action': np.zeros(action_shape, dtype=action_dtype),
        'reward': np.zeros((capacity, 1), dtype=np.float16),
        'next_state': np.zeros((capacity, *state_space), dtype=state_dtype),
        'done': np.zeros((capacity, 1), dtype=np.bool),
        'steps': np.zeros((capacity, 1), dtype=np.uint8)
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

def copy_buffer(dest_buffer, dest_start, dest_end, orig_buffer, orig_start, orig_end, dest_keys=True):
    assert_colorize(dest_end - dest_start == orig_end - orig_start, 
                    'Inconsistent lengths of dest_buffer and orig_buffer.')
    if dest_end - dest_start == 0:
        return
    
    for key in (dest_buffer if dest_keys else orig_buffer).keys():
        dest_buffer[key][dest_start: dest_end] = orig_buffer[key][orig_start: orig_end]
