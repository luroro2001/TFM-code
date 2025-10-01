import numpy as np

def normalize(x, xmin=None, xmax=None, axis=0, pct=5.0):
    if xmin is None:
        xmin = np.percentile(x, pct, axis=axis, keepdims=True)
    if xmax is None:
        xmax = np.percentile(x, 100.0-pct, axis=axis, keepdims=True)
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0,  xmin, xmax

def denormalize(x, xmin, xmax):
    return 0.5 * (x + 1.0) * (xmax - xmin) + xmin
