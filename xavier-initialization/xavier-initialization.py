import numpy as np
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    fan_in = np.asanyarray(fan_in, dtype=float)
    fan_out = np.asanyarray(fan_out, dtype=float)
    W = np.asanyarray(W, dtype=float)
    L = np.sqrt(6 / (fan_in + fan_out))
    out = (W * 2* L) - L
    return out