import numpy as np

# L: This defines a symmetric logarithmic transform and its inverse

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

def inv_symlog(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

# NOTA: The problem with log(x) is that it only works with positive numbers and 
# diverges near zero.
# symlog handles both positives and negatives, compresses large magnitudes
# logarithmically, and keeps small values almost linear