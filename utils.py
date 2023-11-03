import numpy as np


def min_max_normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    
    normalized_x = (x - x_min) / (x_max - x_min)
    
    return normalized_x, x_min, x_max