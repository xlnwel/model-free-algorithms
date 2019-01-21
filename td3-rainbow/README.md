# An Implementation of DDPG in conjunction with Rainbow and others

In which I try to combine DDPG with "Rainbow: Combining Improvements in Deep Reinforcement Learning" and "Twin Delayed Deep Deterministic policy gradient algorithm "

- [x]: Noisy Nets in Actor
- [x]: Multi-Step Learning
- [x]: Twin Delayed Deep Deterministic policy gradient algorithm 
- [x]: Prioritized Replay

## Playground

If you'd like to play around with this project, feel free to tune the arguments in args.yaml.

## Performance

Although Distributional Critic has not been fully functional, this project has already reached the solved requirement in BipedalWalker-v2:

By setting `n_steps=3` and `double=True` in *args.yaml*, we will have the results (I set batch_size to 32 since I run this project on a machine without CUDA gpu. The process could be sped up by a larger batch_size)

Average scores per 100 episodes shown in command line:
<p align="center">
<img src="/results/1.png" alt="average score in command line" height="350">
</p>

Average scores per 100 episodes recorded by tensorboard
<p align="center">
<img src="/results/2.png" alt="average score in tensorboard" height="350">
</p>

Instant scores for each episode recorded by tensorboard
<p align="center">
<img src="/results/3.png" alt="instant score in tensorboard" height="350">
</p>

## References

Timothy P. Lillicrap et al. Continuous Control with Deep Reinforcement Learning

Matteo Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning

Marc G. Bellemare et al. A Distributional Perspective on Reinforcement Learning

Hado van Hasselt et al. Deep Reinforcement Learning with Double Q-Learning

Tom Schaul et al. Prioritized Experience Replay

Meire Fortunato et al. Noisy Networks For Exploration

Scott Fujimoto et al. Addressing Function Approximation Error in Actor-Critic Methods