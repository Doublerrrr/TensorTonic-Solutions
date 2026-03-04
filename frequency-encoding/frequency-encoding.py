import numpy as np
def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    arr = np.asanyarray(values)
    total_samples = arr.size
    if total_samples == 0:
        return np.array([], dtype=float)
    uniques, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    freq_map = counts.astype(float) / total_samples
    result = freq_map[inverse]
    return result.tolist()