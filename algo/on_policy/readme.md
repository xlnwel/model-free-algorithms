## Performance on BipedalWalker-v2

#### PPO with NAE, x-axis denotes time.
<p align="center">
<img src="/results/ppo/time.png" alt="average score in tensorboard" height="350">
</p>

## Experimental Results 

### PPO

PPO requires a significantly larger batch size than off-policy algorithms for a stable learning process and better final results. This may corresponds to the way advantages are computed(both GAE and NAE use the whole return, resulting in high variance), and could be alleviated using TD residuals.

It is important to monitor the value of the approximate KL divergence when fine-tuning PPO. As the paper suggests, small KL divergence oftentimes gives a stable learning process, but a tiny KL divergence may impair the final performance (during my experiment, 0.01 is a good threshold). Furthermore, there is a positive correlation between the KL divergence and clip ratio.

Clipping target value helps.

Masking out states and rewards after the agent encounters a done significantly speeds up the learning.

GAE works better with a smaller learning rate (1e-4), NAE works better with a larger learning rate (3e-4).

NAE is more stable than GAE in my test. Both take around 4 hours to reach 250+ scores averaged over 10 episodes, but take about 8-10 hours to reach the solved requirement of BipedalWalker-v2.