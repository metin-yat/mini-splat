import torch
import numpy as np

# Graphics utilities used by the rendering pipeline.
# Only keep functions that are referenced in the codebase.

def quaternion_to_rotation_matrix(q):
    """
    Convert batched quaternions (w, x, y, z) to 3x3 rotation matrices.

    Args:
      q: tensor of shape (N, 4)

    Returns:
      Tensor of shape (N, 3, 3) with rotation matrices.
    """
    norm = torch.sqrt(q[:, 0]*q[:, 0] + q[:, 1]*q[:, 1] + q[:, 2]*q[:, 2] + q[:, 3]*q[:, 3])
    q = q / norm[:, None]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    res = torch.zeros((q.shape[0], 3, 3), device=q.device)

    res[:, 0, 0] = 1 - 2 * (y*y + z*z)
    res[:, 0, 1] = 2 * (x*y - w*z)
    res[:, 0, 2] = 2 * (x*z + w*y)
    res[:, 1, 0] = 2 * (x*y + w*z)
    res[:, 1, 1] = 1 - 2 * (x*x + z*z)
    res[:, 1, 2] = 2 * (y*z - w*x)
    res[:, 2, 0] = 2 * (x*z - w*y)
    res[:, 2, 1] = 2 * (y*z + w*x)
    res[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return res