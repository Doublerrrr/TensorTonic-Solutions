import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    tp = np.sum(y_true == y_pred)
    total_errors = np.sum(y_true != y_pred)
    fp = total_errors
    fn = total_errors

    precision_denom = tp + fp
    recall_denom = tp + fn
    precision = tp / precision_denom if precision_denom > 0 else 0.0
    recall = tp / recall_denom if recall_denom > 0 else 0.0
    f1_denom = precision + recall
    if f1_denom == 0 :
        return 0.0
    f1 = 2 * (precision * recall) / f1_denom
    return float(f1)
    pass