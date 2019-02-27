import os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility.yaml_op import load_args
from td3_rainbow.replay.select_buffer import construct_buffer


def add_exps(buffer, state, items=1):
    for i in range(items):
        state = state + 1
        action = np.random.randint(5, 10)
        reward = np.random.randint(2, 4)
        done = 0 if np.random.randn() < .9 else 1
            
        buffer.add(state, action, reward, state+1, done)
        if done and len(buffer.temporary_buffer) > 0:
            print('done occurs but temporary buffer is not cleared ')
    return state

def main():
    args = load_args('distributed_args.yaml')
    env_args = args['env']
    agent_args = args['agent']
    buffer_args = args['buffer']

    env = gym.make(env_args['name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    buffer = construct_buffer(buffer_args)
    sample_types = (tf.float32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    sample_shapes =((batch_size), (
        (None, state_dim),
        (None, action_dim),
        (None, 3, 1),
        (None, state_dim),
        (None, 1),
        (None, 1)
    ))
    ds = tf.data.Dataset.from_generator(buffer, sample_types, sample_shapes)
    ds = ds.prefetch(1)
    IS_ratio, (obs, action, reward, next_obs, done, steps) = ds.make_one_shot_iterator().get_next()

    add_exps(buffer, 0, 100)
    for i in range(1000):
        i = add_exps(buffer, i)
        print(i)
        with tf.Session() as sess:
            data = sess.run(obs)
            print(data)
        x = tf.layers.dense(obs, 1)
        with tf.Session() as sess:
            print(sess.run(x))

    for item in buffer.memory:
        print(item)

if __name__ == '__main__':
    main()

