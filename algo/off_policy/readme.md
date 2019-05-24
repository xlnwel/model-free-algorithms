## Performance on BipedalWalker-v2

Here we demonstrate the average score per 100 episodes shown in tensorboard:

#### TD3, learning in a background thread, x-axis denotes running time, three random seeds

<p align="center">
<img src="/results/td3/back_time.png" alt="average score in tensorboard" height="350">
</p>

#### TD3, learning in a background thread, x-axis denotes episodes, three random seeds

<p align="center">
<img src="/results/td3/back_episode.png" alt="average score in tensorboard" height="350">
</p>

#### TD3, learning in the foreground thread, x-axis denotes episodes, three random seeds

<p align="center">
<img src="/results/td3/fore_episode.png" alt="average score in tensorboard" height="350">
</p>

#### SAC, learning in the background thread, x-axis denotes episodes, three random seeds

<p align="center">
<img src="/results/sac/back_time.png" alt="average score in tensorboard" height="350">
</p>

#### SAC, learning in the foreground thread, x-axis denotes episodes, three random seeds

<p align="center">
<img src="/results/sac/fore_episode.png" alt="average score in tensorboard" height="350">
</p>

#### Ape-X, 10 workers, bach size 512 x-axis denotes time
## Implementation Details

### Some Implementation Details

1. All off-policy algorithms use proportional replay as the default experience replay buffer, and noisy layers as the exploration strategy.

2. Unlike many open source RL algorithms, network update is by default run in a background thread here. 
    The advantage is:

    - This speeds up the learning process. (2x+)

    The downside is:

    - This makes the algorithm unstable at the convergence, which, however, could be mitigated by learning rate decay and a large batch size.
    - This causes loss of control of the number of updates per environment step, which is sometimes required when doing research experiments, but we do not concern it here.256 

## Experimental Results

## Common

Large networks are easy to result in fluctuation at the convergence, which may be caused by overfitting.

Add actions at the first two levels improve performance. This effect is more significant when having learning in a background thread.

Large networks slow down the learning process, and worse still, may impair the final performance, resulting in fluctuation at the convergence. This may be the result of overfitting.

Training networks in a background thread may cause learning unstable at the convergence, which could be alleviated by increasing the batch size.

Adding noisy layers at the last two dense layers significantly helps do exploration.

### TD3

Applying bias correction for prioritized sampling to actor loss improves the performance.

### SAC

It is better to use Q-error instead of V-error as priority

It is hard to tell the effect of noisy layers in SAC. During experiments noisy layers in deed speed up the learning process. 

Bias correction for prioritized sampling helps.

### Ape-X

Ape-X requires a larger batch size than general single-process algorithms.