import numpy as np


def reset_buffer(buffer, capacity, state_dim, action_dim, has_priority):
    target_buffer = {'priority': np.zeros((capacity, 1))} if has_priority else {}
    target_buffer.update({
        'state': np.zeros((capacity, state_dim)),
        'action': np.zeros((capacity, action_dim)),
        'reward': np.zeros((capacity, 1)),
        'next_state': np.zeros((capacity, state_dim)),
        'done': np.zeros((capacity, 1)),
        'steps': np.zeros((capacity, 1))
    })

    buffer.update(target_buffer)

def add_buffer(buffer, idx, state, action, reward, next_state, done, n_steps, gamma):
    buffer['state'][idx] = state
    buffer['action'][idx] = action
    buffer['reward'][idx] = reward
    buffer['next_state'][idx] = next_state
    buffer['done'][idx] = done
    buffer['steps'][idx] = 1
    for i in range(1, n_steps):
        k = idx - i
        if buffer['done'][k] == True:
            break
        buffer['reward'][k] += gamma**i * reward
        buffer['next_state'][k] = next_state
        buffer['done'][k] = done
        buffer['steps'][k] += 1

def copy_buffer(dest_buffer, dest_start, dest_end, orig_buffer, orig_start, orig_end):
    assert dest_end - dest_start == orig_end - orig_start, 'Inconsistent lengths of dest_buffer and orig_buffer.'
    if dest_end - dest_start == 0:
        return
    
    dest_buffer['state'][dest_start: dest_end] = orig_buffer['state'][orig_start: orig_end]
    dest_buffer['action'][dest_start: dest_end] = orig_buffer['action'][orig_start: orig_end]
    dest_buffer['reward'][dest_start: dest_end] = orig_buffer['reward'][orig_start: orig_end]
    dest_buffer['next_state'][dest_start: dest_end] = orig_buffer['next_state'][orig_start: orig_end]
    dest_buffer['done'][dest_start: dest_end] = orig_buffer['done'][orig_start: orig_end]
    dest_buffer['steps'][dest_start: dest_end] = orig_buffer['steps'][orig_start: orig_end]
