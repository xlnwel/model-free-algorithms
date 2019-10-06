## Performance on BipedalWalker-v2

#### Learning curve for TD3 and SAC, x-axis denotes episodes, y-axis denotes episodic reward averaged over 100 episodes at training time, three random seeds

Here we demonstrate the learning curve **at training time**. One could conceives that evaluation will give significantly better results since y-axis denotes episodic reward average over the latest 100 episodes:

<p align="center">
<img src="/results/td3-sac.png" alt="td3-sac" height="350">
</p>

Our implementations significantly boost learning process, steadily solving [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) in 200-400 episodes compared to the best implementation in the [Leaderboard](https://github.com/openai/gym/wiki/Leaderboard#bipedalwalker-v2), which occasionally solves it in 800 episodes.

## Implementation Details

### Some Implementation Details

All off-policy algorithms use proportional replay as the default experience replay buffer, and noisy layers as the exploration strategy.

## Experimental Results

## Common

Adding actions at the first two levels improve performance. 

Large networks slow down the learning process, and worse still, may impair the final performance, resulting in fluctuation at the convergence. This may be the result of overfitting.

Adding noisy layers at the last two dense layers significantly helps exploration.

### Rainbow-IQN

Double DQN, Deuling DQN, IQN are implemented and tested on CartPole-v0 and LunarLander-v2 from OpenAI's GYM. 

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