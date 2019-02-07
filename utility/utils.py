import numpy as np

def normalize(x, mean=0., std=1., epsilon=1e-8):
    x = (x - np.mean(x)) / (np.std(x) + epsilon)
    x = x * std + mean

    return x

def schedule(start_value, step, decay_steps, decay_rate):
    return start_value * decay_rate**(step // decay_steps)