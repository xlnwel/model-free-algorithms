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

## Implementation Details

### Some Implementation Details

All off-policy algorithms use proportional replay as the default experience replay buffer, and noisy layers as the exploration strategy.

## Experimental Results

## Common

Large networks are easy to result in chattering at the convergence, which may be caused by overfitting.

Adding actions at the first two levels improve performance. This effect is more significant when having learning in a background thread.

Large networks slow down the learning process, and worse still, may impair the final performance, resulting in fluctuation at the convergence. This may be the result of overfitting.

Training networks in a background thread may cause learning unstable at the convergence, which could be alleviated by increasing the batch size.

Adding noisy layers at the last two dense layers significantly helps to do exploration.

### Rainbow-IQN

Double DQN, Deuling DQN, IQN are implemented and tested on CartPole-v0 and LunarLander-v2 from OpenAI's GYM. Algorithms could be specified in [args.yaml](https://github.com/xlnwel/model-free-algorithms/blob/master/algo/off_policy/rainbow_iqn/args.yaml)(by changing the value of `algo` in `Qnets`.

Distributional DQN(aka., c51) is not included since it is extremely hard to fine-tune, and IQN could be a perfect replacement of it.

For these algorithms on atari, please refer to my [another project](https://github.com/xlnwel/atari_rl)

### TD3

Applying bias correction for prioritized sampling to actor loss improves the performance.

### SAC

It is better to use Q-error instead of V-error as priority

It is hard to tell the effect of noisy layers in SAC. During experiments noisy layers in deed speed up the learning process. 

Bias correction for prioritized sampling helps.

### Ape-X

Ape-X requires a larger batch size than general single-process algorithms.

Do not use noisy layers in Apex-X with SAC.