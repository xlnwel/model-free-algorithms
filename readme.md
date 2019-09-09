## Algorithms Implemented

- [x] TD3       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy/td3)]
- [x] SAC       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy/sac)]
- [x] IQN       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy/rainbow_iqn)]
- [x] Rainbow   [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy/rainbow_iqn)]
- [x] PPO       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/on_policy/ppo)]
- [x] A2C       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/on_policy/a2c)]
- [x] Apex      [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy/apex)]
- [x] Noisy Nets[[code](https://github.com/xlnwel/model-free-algorithms/blob/b471f32c1ecc15632da097cf150bc8999a314aa9/basic_model/layer.py#L193)]
- [x] PER       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy/replay)]
- [x] GAE       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/on_policy/ppo)]
- [x] NAE       [[code](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/on_policy/ppo)]

## Overall Architecture

<p align="center">
<img src="/results/Architecture.png" alt="Architectureg" height="650">
</p>

Algorithms are implemented in [algo](https://github.com/xlnwel/model-free-algorithms/tree/master/algo)

## Notes

All of these algorithms are implemented using the same low-level API. This may cause trouble to some early-implemented algorithms due to my obliviousness as the low-level API evolves from time to time. If you happen to bump into one, feel free to open an issue. 

Distributed Algorithms are implemented using [Ray](https://ray.readthedocs.io/en/latest/), a flexible, high-performance distributed execution framework.

Due to the lack of a Mujoco license, all algorithms for continuous control are tested on the [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2) and [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) environments from OpenAI's Gym and solve them. In particular, our TD3 and SAC solve BipedalWalker-v2 in 2-4 hours, significantly faster than the best one on the [Leaderboard](https://github.com/openai/gym/wiki/Leaderboard#bipedalwalker-v2). On the other hand, PPO, which runs in 32-environment vector, steadily solves it in 5-8 hours. 

Rainbow, IQN is tested on CartPole-v0 and and steadily solves it. For Rainbow and IQN on Atari games, please refer to my [another project](https://github.com/xlnwel/atari_rl)

Performance figures and some further experimental results are recorded in [on-policy algorithms](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/on_policy) and [off-policy algorithms](https://github.com/xlnwel/model-free-algorithms/tree/master/algo/off_policy).

Best arguments are kept in "args.yaml" in each algorithm folder. If you want to modify some arguments, do not modify it in "args.yaml". It is better to first pass the experimental arguments to `gs` defined in [run/train.py](https://github.com/xlnwel/model-free-algorithms/blob/master/run/train.py) to verify that they do improve the performance.

## Requirements

It is recommended to install Tensorflow from source following [this instruction](https://www.tensorflow.org/install/source) to gain some CPU boost and other potential benefits.

```shell
# Minimal requirements to run the algorithms. Tested on Ubuntu 18.04.2, using Tensorflow 1.13.1.
conda create -n gym python
conda activate gym
pip install -r requirements.txt
# Install tensorflow-gpu or install it from scratch as the above instruction suggests
pip install tensorflow-gpu
```

## Running

```shell
# Silence tensorflow debug message
export TF_CPP_MIN_LOG_LEVEL=3
# When running distributed algorithms, restrict numpy to one core
# Use numpy.__config__.show() to ensure your numpy is using OpenBlas
# For MKL and detailed reasoning, refer to [this instruction](https://ray.readthedocs.io/en/latest/example-rl-pong.html?highlight=openblas#the-distributed-version)
export OPENBLAS_NUM_THREADS=1

# For full argument specification, please refer to run/train.py
python run/train.py -a=td3
```
To add monitor so as to save videos automatically, add argument `video_path` to `env` in `args.yaml`. See `env` in `algo/off_policy_td3/args.yaml` for a concrete example. 

## Paper References

Timothy P. Lillicrap et al. Continuous Control with Deep Reinforcement Learning

Matteo Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning

Marc G. Bellemare et al. A Distributional Perspective on Reinforcement Learning

Scott Fujimoto et al. Addressing Function Approximation Error in Actor-Critic Methods (TD3)

Tuomas Haarnoja et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

Dan Horgan et al. Distributed Prioritized Experience Replay 

Hado van Hasselt et al. Deep Reinforcement Learning with Double Q-Learning

Tom Schaul et al. Prioritized Experience Replay

Meire Fortunato et al. Noisy Networks For Exploration

Ziyu Wang et la. Dueling Network Architectures for Deep Reinforcement Learning

Will Dabney et al. Implicit Quantile Networks for Distributional Reinforcement Learning

Berkeley cs294-112

## Code References

OpenAI Baselines

[Homework of Berkeley CS291-112](http://rail.eecs.berkeley.edu/deeprlcourse/)

[Google Dopamine](https://github.com/google/dopamine)