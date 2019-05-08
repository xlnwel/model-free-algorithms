## Heads Up

a2c is legacy code and no longer maintained because I personally think a2c could be completely replaced by using an environment vector if one only intends to run ppo on a single machine.

## Experimental Results 

### PPO

PPO requires a significantly large batch size than Off-policy algorithm for a stable learning process and better final results.

It is important to monitor the value of the approximate KL divergence when fine-tuning PPO. As the paper suggests, small KL divergence oftentimes gives a stable learning process, but a tiny KL divergence may impair the final performance (during my experiment, 0.01 is a good threshold). Furthermore, there is a positive correlation between the KL divergence and clip ratio.

Clipping target value helps.

Masking out states, actions, and rewards after done helps. THe choice of masking really matters.

gae advantage should work with lower learning rate, but I do not spot that it outperforms norm advantage