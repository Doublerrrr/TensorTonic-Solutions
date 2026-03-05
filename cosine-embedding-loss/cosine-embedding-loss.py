import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    x1 = np.asanyarray(x1, dtype=float)
    x2 = np.asanyarray(x2, dtype=float)
    y = np.asanyarray(label)

    norm1 = np.linalg.norm(x1, axis=-1)
    norm2 = np.linalg.norm(x2, axis=-1)

    dot_product = np.sum(x1*x2, axis=-1)
    cos_sim = dot_product / (norm1*norm2 + 1e-15)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    loss_pos = 1.0 - cos_sim
    loss_neg = np.maximum(0.0, cos_sim - margin)
    total_loss = np.where(y==1, loss_pos, loss_neg)
    return float(np.mean(total_loss))