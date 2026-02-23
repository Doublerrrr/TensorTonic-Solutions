import numpy as np
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X = np.asanyarray(X)
    W = np.asanyarray(W)
    b = np.asanyarray(b)
    Z = (X @ W) + b
    return Z.tolist()