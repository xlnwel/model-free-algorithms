from collections import namedtuple, deque
import numpy as np

class ReplayBuffer():
    def __init__(self, args, sample_size, n_steps=1, gamma=1):
        self.capacity = args['capacity'] if 'capacity' in args else 10
        self.min_size = args['min_size'] if 'min_size' in args else 5
        self.memory = None
        self.sample_size = sample_size
        # rewards here are discounted
        # 'next_state' here is the state n_steps latter
        self.experience = namedtuple('experience', ('state', 'action', 'rewards', 'next_state', 'done', 'steps'))

        self.n_steps = n_steps
        self.gamma = gamma
        if self.n_steps < 1:
            raise ValueError('n_steps requires positive integer!')
        elif self.n_steps > 1:
            # temporary buffer for imcomplete experience
            self.temporary_buffer = deque(maxlen=self.n_steps)

        self.full = False

    @property
    def good_to_learn(self):
        return len(self) >= self.min_size

    def add(self, state, action, reward, next_state, done):
        rewards = np.zeros(self.n_steps)
        exp = self.experience(state, action, rewards, next_state, done, 1)

        if self.n_steps > 1:
            original_n = len(self.temporary_buffer)

            self.temporary_buffer.append(exp)

            # add discounted reward to each exp in temporary_buffer
            for i, e in enumerate(self.temporary_buffer):
                # WARNING: exp as tuple here will introduce some overhead
                steps = original_n - i + 1
                e.rewards[steps - 1] = self.gamma ** (steps - 1) * reward
                self.temporary_buffer[i] = self.experience(e.state, e.action, e.rewards, next_state, done, steps)
                
            if done:
                # flush all elements in buffer to memory if the episode is done
                for e in self.temporary_buffer:
                    self._add_exp(e)
                self.temporary_buffer.clear()
            elif len(self.temporary_buffer) == self.temporary_buffer.maxlen:
                # add the ready exp in the buffer to memory
                ready_exp = self.temporary_buffer.popleft()
                self._add_exp(ready_exp)
        elif self.n_steps == 1:
            exp.rewards[0] = reward
            self._add_exp(exp)
        else:
            raise ValueError('Invalid n for multi-step learning')

        # Test when memory if full
        if not self.full and len(self) == self.capacity:
            print('\nfull memory')
            self.full = True

    def sample(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # Implementation
    def _add_exp(self, exp):
        raise NotImplementedError

    def _unpack_samples(self, exps):
        states = np.vstack([e.state for e in exps if e is not None])
        actions = np.vstack([e.action for e in exps if e is not None])
        rewards = np.expand_dims(np.vstack([e.rewards for e in exps if e is not None]), axis=-1)
        next_states = np.vstack([e.next_state for e in exps if e is not None])
        dones = np.vstack([e.done for e in exps if e is not None])
        steps = np.vstack([e.steps for e in exps if e is not None])

        return states, actions, rewards, next_states, dones, steps
