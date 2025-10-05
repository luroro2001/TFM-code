import numpy as np

def normalize(x, xmin=None, xmax=None, axis=0, pct=5.0):
    """
    L:
    Takes an input array x and scales it to [-1,1]. If the scaling bounds are not
    provided, they are estimated from percentiles
    """
    if xmin is None:
        #lower bound=5th percentile (by default)
        xmin = np.percentile(x, pct, axis=axis, keepdims=True)
    if xmax is None:
        #upper bound=95th percentile (by default)
        xmax = np.percentile(x, 100.0-pct, axis=axis, keepdims=True)
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0,  xmin, xmax #returns normalization and bounds

def denormalize(x, xmin, xmax):
    """
    L:
    Inverts the normalization: takes a value in [-1,1] and maps it to the original scale
    """
    return 0.5 * (x + 1.0) * (xmax - xmin) + xmin
