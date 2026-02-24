import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    P = np.asanyarray(points, dtype=float)
    A = np.asanyarray(assignments)
    n_features = P.shape[1]
    new_centroids = np.zeros((k, n_features))
    for i in range(k):
        cluster_members = P[A == i]
        if cluster_members.shape[0] > 0:
            new_centroids[i] = cluster_members.mean(axis=0)
        else:
            pass
    return new_centroids.tolist()