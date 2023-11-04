import numpy as np

def min_max_normalize(x):
    x_min, x_max = np.min(x), np.max(x)
    normalized_x = (x - x_min) / (x_max - x_min)
    return normalized_x, x_min, x_max
