## Performance on BipedalWalker-v2

#### Learning curve for TD3 and SAC, x-axis denotes episodes, y-axis denotes episodic reward averaged over 100 episodes at training time, three random seeds

Here we demonstrate the learning curve **at training time**(from commit f4359cc4655b08250592b9c37f57965d9bd0329b). One could conceives that evaluation will give significantly better results since y-axis denotes episodic reward average over the latest 100 episodes:

<p align="center">
<img src="/results/td3-sac.png" alt="td3-sac" height="350">
</p>

Our implementations significantly boost learning process, steadily solving BipedalWalker-v2 in 200-400 episodes compared to the best implementation in the OpenAI's Leaderboard, which occasionally solves it in 800 episodes. 

### Learning curve of Apex-TD3

<p align="center">
<img src="/results/apex-td3.png" alt="apex-td3" height="350">
</p>


## Experimental Observation

The following observations are based on results of [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) averaged over three random seeds

### Common

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

2. Adaptvie temporature based on state-action performs best in practice.

### TD3

1. TD3 does not work well with reward clipping and reward scaling. In fact, plain reward works fine with TD3
