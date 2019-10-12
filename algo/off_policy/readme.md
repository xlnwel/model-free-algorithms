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

## Experimental Observation

### Common

The following observations are based on results of [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) averaged over three random seeds,

1. Adding actions to the first two layers. 

2. Adding noisy layers to the last two dense layers.

3. Layer normalization.

4. Reward normalization and scaling. We also truncate the terminal reward. -100 is too strong, causing the running reward statistics shift. It is noteworthy that truncating the reward only helps in the case we normalize the rewards; it hurts the performance if rewards are not normalized.

5. Although small networks are sufficient to solve BepedalWalker-v2, large networks speed up the learning process, and often result in more stable final results. Here large network suggests a deeper one --- simply increasing hidden units does not help much. We suspect this has something to do with the selection of noisy layers.

6. Shut down noisy layers at evaluation significantly impair the performance

7. We test three forms of state normalization: 1) normalize with statistics computed from presampled states, 2) normalize with running stata statistics, and 3) instance normalization. None of them look promising.

8. Learning rate annealing

9. Action repetition

### SAC

1. Value function 

2. Adaptvie temporature based on state-action performs best in practice.
