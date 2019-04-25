``` shell
conda create -n gym python=3.6
pip install -r requirements.yml
```

```shell
# train
source activate gym

# silence tensorflow debug message
export TF_CPP_MIN_LOG_LEVEL=3
# For distributed training using Ray
# For more details: https://ray.readthedocs.io/en/latest/example-rl-pong.html?highlight=openblas
export OPENBLAS_NUM_THREADS=1

# for full argument specification, please refer to run/train.py
python run/train.py -a=td3
```

## Algorithms Implemented

- [x]: Noisy Nets in Actor
- [x]: Multi-Step Learning
- [x]: PER
- [x]: TD3
- [x]: PPO
- [x]: A3C
- [x]: A2C
- [x]: Apex

## Some Common Implementation Details

All off-policy algorithms use proportional replay as default experience replay buffer.
Noisy nets are used as default exploration strategy.


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

Berkeley CS291-112