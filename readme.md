## Algorithms Implemented

- [x]: Noisy Nets
- [x]: PER
- [x]: TD3
- [x]: PPO
- [x]: A3C
- [x]: A2C
- [x]: Apex

## Notes

Distributed Algorithms are implemented using [Ray](https://ray.readthedocs.io/en/latest/), a flexible, high-performance distributed execution framework.

Experimental results are recorded in `algo/on_policy/readme.md` and `algo/off_policy/readme.md`.

All code are tested in [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/) first, and then [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/). The later is a pretty challenging game, 

Best arguments are kept in `args.yaml` in each algorithm folder.

## Running

```shell
# silence tensorflow debug message
export TF_CPP_MIN_LOG_LEVEL=3
# For distributed training using Ray
# For more details: https://ray.readthedocs.io/en/latest/example-rl-pong.html?highlight=openblas
export OPENBLAS_NUM_THREADS=1

# for full argument specification, please refer to run/train.py
python run/train.py -a=td3
```

## Paper References

Timothy P. Lillicrap et al. Continuous Control with Deep Reinforcement Learning

Matteo Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning

Marc G. Bellemare et al. A Distributional Perspective on Reinforcement Learning

Hado van Hasselt et al. Deep Reinforcement Learning with Double Q-Learning

Tom Schaul et al. Prioritized Experience Replay

Meire Fortunato et al. Noisy Networks For Exploration

Scott Fujimoto et al. Addressing Function Approximation Error in Actor-Critic Methods (TD3)

Tuomas Haarnoja et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

Dan Horgan et al. Distributed Prioritized Experience Replay 

Berkeley cs294-112

## Code References

OpenAI Baselines

OpenAI Spinning UP

Homework of Berkeley CS291-112