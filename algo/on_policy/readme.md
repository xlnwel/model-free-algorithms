## Heads Up

a2c is legacy code and no longer maintained because I personally think a2c could be completely replaced by using an environment vector if one only intends to run ppo on a single machine.

## Performance on BipedalWalker-v2

<p align="center">
<img src="/results/ppo/time.png" alt="average score in tensorboard" height="350">
</p>

## Experimental Results 

### PPO

PPO requires a significantly large batch size than Off-policy algorithm for a stable learning process and better final results.

It is important to monitor the value of the approximate KL divergence when fine-tuning PPO. As the paper suggests, small KL divergence oftentimes gives a stable learning process, but a tiny KL divergence may impair the final performance (during my experiment, 0.01 is a good threshold). Furthermore, there is a positive correlation between the KL divergence and clip ratio.

Clipping target value helps.

To my surprise, mask out states and rewards after the agent encounters a done does not improve performance

GAE works better with a smaller learning rate (1e-4), NAE works better with a larger learning rate (3e-4).

NAE is more stable than GAE. Both take around 4 hours to reach 250+ scores averaged over 10 episodes, but take about 8-10 hours to solve BipedalWalker-v2.