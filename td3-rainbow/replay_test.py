import numpy as np
from replay.proportional_replay import ProportionalPrioritizedReplay
from replay.rank_based_replay import RankBasedPrioritizedReplay

sample_size = 5

def print_container(buffer):
    for i in range(len(buffer)):
        print(i, buffer.data_structure.container[i])
        
def build_buffer(name, n_steps, gamma):
    if name == 'rank':
        buffer = RankBasedPrioritizedReplay(args, sample_size, n_steps, gamma)
    else:
        buffer = ProportionalPrioritizedReplay(args, sample_size, n_steps, gamma)

    return buffer

def add_exps(buffer, extra_items):
    start_state = np.random.randint(10)
    for i in range(buffer.capacity + extra_items):
        action = np.random.randint(5, 10)
        reward = np.random.randint(2, 4)
        done = 0 if np.random.randn() < .9 else 1
            
        buffer.add(i, action, reward, i+1, done)
        if done and len(buffer.temporary_buffer) > 0:
            print('done occurs but temporary buffer is not cleared ')


def test_add_to_memory(buffer, name, n_steps, gamma, extra_items):
    """ if nothing is printed, add exp to buffer.memory should work fine """
    # test correctness of exps in memory
    for k in range(1, buffer.n_steps):
        for i in range(len(buffer)):
            if buffer.memory[i].done:
                continue
            if i == (extra_items - len(buffer.temporary_buffer) - 1)%buffer.capacity:
                continue
            # if the following code print something, the reward computation is wrong
            if buffer.memory[i].rewards[k] != buffer.gamma * buffer.memory[(i+1) % len(buffer)].rewards[k-1]:
                print('name:', name, 'n_steps:', n_steps, 'gamma:', gamma, 'extra_items:', extra_items, 'Ops!!!')
                print(i, k)
                for exp in buffer.memory:
                    print(exp)

def test_sum_tree_consistency(buffer, n_steps, gamma, extra_items):
    sum_tree = buffer.data_structure

    for i in range(buffer.capacity-1):
        left, right = 2 * i + 1, 2 * i + 2
        if sum_tree.container[i][0] != sum_tree.container[left][0] + sum_tree.container[right][0]:
            print('name:', 'prop', 'n_steps:', n_steps, 'gamma:', gamma, 'extra_items:', extra_items, 'Ops!!!')
            print('sum tree is not consistent')

def test_update_priorities_prop(buffer, n_steps, gamma, extra_items):
    _, _ = buffer.sample()
    
    priorities = np.random.normal(loc=1, scale=1, size=sample_size)

    buffer.update_priorities(priorities)
    
    test_sum_tree_consistency(buffer, n_steps, gamma, extra_items)

def test_update_priorities_rank(buffer):
    _, _ = buffer.sample()
    print('Test update rank priority')
    print_container(buffer)
    priorities = np.random.normal(loc=1, scale=1, size=sample_size)
    print('saved exp ids')
    print(buffer._saved_exp_ids)
    print('priorities', priorities)
    buffer.update_priorities(priorities)
    print('priorities updated')
    print_container(buffer)

if __name__ == "__main__":
    args = {}

    for n_steps in [3, 5]:
        for gamma in [0, .5, 1]:
            for extra_items in [0, 7, 23]:
                for name in ['rank']:
                    print('name:', name, 'n_steps:', n_steps, 'gamma:', gamma, 'extra_items:', extra_items)
                    buffer = build_buffer(name, n_steps, gamma)
                    add_exps(buffer, extra_items)
                    test_add_to_memory(buffer, name, n_steps, gamma, extra_items)

                    if name == 'rank':
                        test_update_priorities_rank(buffer)
                    else:
                        test_sum_tree_consistency(buffer, n_steps, gamma, extra_items)
                        test_update_priorities_prop(buffer, n_steps, gamma, extra_items)
