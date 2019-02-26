import os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility.yaml_op import load_args
from td3_rainbow.replay.select_buffer import construct_buffer


def add_exps(buffer, state, extra_items=0):
    for i in range(buffer.capacity + extra_items):
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

    buffer = construct_buffer(buffer_args)
    ds = tf.data.Dataset.from_generator(buffer, (tf.float32, 
        (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)))
    val = ds.make_one_shot_iterator().get_next()
    i = 1
    i = add_exps(buffer, i)
    i = add_exps(buffer, i)

    for i in range(3):
        print(i)
        with tf.Session() as sess:
            data = sess.run(val)
            print(data)

    for item in buffer.memory:
        print(item)

if __name__ == '__main__':
    main()

