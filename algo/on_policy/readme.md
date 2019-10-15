## Performance on BipedalWalker-v2

#### PPO, x-axis denotes epochs.

<p align="center">
<img src="/results/ppo.png" alt="average score in tensorboard" height="350">
</p>

## Experimental Results 

### PPO

PPO requires a significantly larger batch size than off-policy algorithms for a stable learning process and better final results. This may corresponds to the score function estimator and the way advantages are computed(both GAE and NAE use the whole return, resulting in high variance).

It is important to monitor the value of the approximate KL divergence when fine-tuning PPO. As the paper suggests, small KL divergence oftentimes gives a stable learning process, but a tiny KL divergence may impair the final performance (during my experiment, 0.01 is a good threshold). Furthermore, there is a positive correlation between the KL divergence and clip ratio.

Clipping target value helps.

Comparison between GAE and NAE(based on a single trial):

1. When apply GAE, it is better to update value function more often so that the adantage estimate will be more accurate. However, this is not the case for NAE, the reason is still unclear.

2. GAE works better with a lower max KL bound(0.01); NAE works better with a larger max KL bound(0.02). Overall, I find NAE is more stable than GAE.

3. As expected, GAE works best with both data masking and loss masking applied; Surprising, NAE does not work well with data masking, but works fine with loss masking.

Sum them up: NAE works best for a naive implementation; for some mysterious reasons, it is not compatible with tricks, such as data masking and multiple value updates, which significantly improve the performance of GAE.
