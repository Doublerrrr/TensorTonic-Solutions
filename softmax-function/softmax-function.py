import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x_arr = np.asanyarray(x, dtype=float)
    axis = -1
    x_max = np.max(x_arr, axis = axis, keepdims=True)
    exps = np.exp(x_arr - x_max)
    sum_eps = np.sum(exps, axis=axis, keepdims=True)
    return exps / sum_eps