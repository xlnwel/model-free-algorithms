By default, we use multi-step, PER, noisy layers in all implementations.

## Performance on BipedalWalker-v2

#### Learning curve for TD3 and SAC, x-axis denotes episodes, y-axis denotes episodic reward averaged over 100 episodes at training time, three random seeds

Here we demonstrate the learning curve **at training time**. One could conceives that evaluation will give significantly better results since y-axis denotes episodic reward average over the latest 100 episodes:

<p align="center">
<img src="/results/td3-sac.png" alt="td3-sac" height="350">
</p>

Our implementations significantly boost learning process, steadily solving BipedalWalker-v2 in 200-400 episodes compared to the best implementation in the OpenAI's Leaderboard, which occasionally solves it in 800 episodes. 

### Learning curve of Apex-TD3

<p align="center">
<img src="/results/apex-td3.png" alt="apex-td3" height="350">
</p>

### Learning curve of SAC

<p align="center">
<img src="/results/apex-sac.png" alt="apex-td3" height="350">
</p>

The first agent takes determinisitc actions.
