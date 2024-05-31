import math
import numpy as np

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-(x - mu)**2 / (2 * sigma**2))