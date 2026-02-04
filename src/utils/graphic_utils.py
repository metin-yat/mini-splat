import torch
import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    q: (N, 4) tensor
    """
    norm = torch.sqrt(q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + q[:,3]*q[:,3])
    q = q / norm[:, None]
    
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    
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

def get_projection_matrix(znear, zfar, fovX, fovY):
    # Projection matrix required to project 3D points onto the 2D screen
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.transpose(0, 1)

def world_to_view(R, t):
    # Creates the World-to-Camera (W2C) matrix
    rt = torch.zeros((4, 4))
    rt[:3, :3] = R.transpose(0, 1)
    rt[:3, 3] = t
    rt[3, 3] = 1.0
    return rt.transpose(0, 1)