## Implementation Details

### Common Details

1. All off-policy algorithms use proportional replay as default experience replay buffer, and noisy layers as exploration strategy.

2. Unlike many open source RL algorithms, network update is by default run in a background thread here. 
    The advantage is:

    - This speeds up the learning process. (2x+)

    The downside is:

    - This makes the algorithm more fluctuate at the convergence, but this could be mitigate through learning rate decay
    - This causes loss of control of the number of updates per environment step, which is sometimes required when doing research experiments, but we do not concern it here. 

## Experimental Results

### SAC

Using Q-error instead of V-error as priority

Learning rate decay helps stabalize the learning at convergence

### TD3

Applying bias correction for prioritized sampling to actor loss improves the performance.

Add actions at the first two level improve performance
