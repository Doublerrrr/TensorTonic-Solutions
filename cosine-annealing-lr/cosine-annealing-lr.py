import numpy as np
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Write code here
    if current_step >= total_steps:
        return float(min_lr)
    progress = current_step / total_steps
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    return float(lr)