import numpy as np
def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    if not values:
        return []
    arr = np.asanyarray(values, dtype=float)
    v_min = arr.min()
    v_max = arr.max()
    if v_min == v_max:
        return [0] * len(values)
    width = (v_max - v_min) / num_bins
    indices = np.floor((arr - v_min) / width).astype(int)
    final_bins = np.clip(indices, 0, num_bins - 1)
    return final_bins.tolist()