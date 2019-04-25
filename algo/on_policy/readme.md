## Heads Up

a2c is legacy code and no longer maintained, since I personally think a2c could be completely replaced by using a environment vector if one is only intend to run ppo on a single machine.

## Experimental Results 

### PPO

PPO requires a significantly large batch size than Off-policy algorithm for stable learning process and better final results.

It is important to monitor the value of the approximate KL divergence when fine-tuning PPO. As the paper suggests, small KL divergence oftentimes gives a stable learning process, but a tiny KL divergence may impair the final performance (during my experiment, 0.01 is a good threshold). Furthermore, there is a positive correlation between KL divergence and clip ratio.

### IMPALA

Different from IMPALA introduced by the paper, I have the worker send its experience only when it has accomplished a whole trajectory. In that case we no longer need to send the initial LSTM states, thereby we avoid the undesirable initial LSTM state lag caused by inconsistent networks of the worker and the learner.