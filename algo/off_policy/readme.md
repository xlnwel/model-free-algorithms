By default, we use multi-step, PER, noisy layers in all implementations.

## BipedalWalkerHardcore-v2 Trained using Apex-SAC

![](/results/BipedalwalkerHardcore-v2.gif)

## Performance on BipedalWalker-v2

#### Learning curve for TD3 and SAC, x-axis denotes episodes, y-axis denotes episodic reward averaged over 100 episodes at training time, three random seeds

Here we demonstrate the learning curve. Evaluation is done after every 100 episodes.

<p align="center">
<img src="/results/td3-sac.png" alt="td3-sac" height="350">
</p>

Our implementations significantly boost learning process, steadily solving BipedalWalker-v2 in 200 episodes. Although the newly-added best implementation in the OpenAI's [Leaderboard](https://github.com/openai/gym/wiki/Leaderboard#BipedalWalker-v2) also solves it in 200 episodes, they [train the agent 1.5 times](https://github.com/createamind/DRL/blob/b03cf3e6aa5a253bf70f84e7d6b008c2465a27a3/spinup/algos/sac1/sac1_BipedalWalker-v2_200ep.py#L331) as many training steps as we do.

### Learning curve of Apex-TD3

<p align="center">
<img src="/results/apex-td3.png" alt="apex-td3" height="350">
</p>

### Learning curve of SAC

<p align="center">
<img src="/results/apex-sac.png" alt="apex-td3" height="350">
</p>

The first agent takes determinisitc actions.
