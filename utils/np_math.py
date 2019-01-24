import numpy as np

def norm(x, mean=0., std=1., epsilon=1e-8):
    normalized_x = (x - np.mean(x)) / (np.std(x) + epsilon)
    x = normalized_x * std + mean

    return x
