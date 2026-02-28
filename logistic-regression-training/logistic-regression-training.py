import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asanyarray(X, dtype=float)
    y = np.asanyarray(y, dtype=float).reshape(-1,1)
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    for i in range(steps):
        z = (X @ w) + b
        y_pred = _sigmoid(z)
        dw = (1.0 / m) * (X.T @ (y_pred - y))
        db = (1.0 / m) * np.sum(y_pred - y)
        w -= lr*dw
        b -=lr*db
    return w.flatten(), float(b)
    pass