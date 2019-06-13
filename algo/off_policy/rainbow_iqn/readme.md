Double DQN, Deuling DQN, IQN are implemented and tested on CartPole-v0 and LunarLander-v2 from OpenAI's GYM. Algorithms could be specified in [args.yaml](https://github.com/xlnwel/model-free-algorithms/blob/master/algo/off_policy/rainbow_iqn/args.yaml)(by changing the value of `algo` in `Qnets`.

Distributional DQN(aka., c51) is not included since it is extremely hard to fine-tune, and IQN could be a perfect replacement of it.

Exploration is achieved using noisy nets by default, and PER is used for experience replay.