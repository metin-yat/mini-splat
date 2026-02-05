# -----------------------------------------------------------------------------
# GaussianModel: a simple PyTorch module for per-point Gaussian parameters 
# (position, color, scale, rotation, opacity) used in rendering and quick experiments.
#
# Usage:
#   - Initialize from point cloud: create_from_pcd(xyz, rgb)
#   - Query derived attributes: get_covariance(), project_gaussians_2d(camera)
#
# Notes:
#   - Designed for easy testing (inputs: xyz (N, 3) and rgb (N, 3))
#   - See test_pcd.py for example use with COLMAP outputs and basic validation.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from .graphics_utils import quaternion_to_rotation_matrix

class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Main parameters that will be updated during training
        self._xyz = nn.Parameter(torch.empty(0))           # Positions (N, 3)
        self._features_dc = nn.Parameter(torch.empty(0))   # Colors (0th Degree SH - RGB) (N, 1, 3)
        self._scaling = nn.Parameter(torch.empty(0))       # Scales (N, 3)
        self._rotation = nn.Parameter(torch.empty(0))      # Rotations (Quaternion) (N, 4)
        self._opacity = nn.Parameter(torch.empty(0))       # Opacity (N, 1)

    @property
    def get_scaling(self):
        return torch.clamp(torch.exp(self._scaling), max=10.0) 

    @property
    def get_rotation(self):
        # Quaternion should be unit-length (normalized)
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        # Opacity should be between 0 and 1
        return torch.sigmoid(self._opacity)

    def get_features(self):
        return self._features_dc

    def create_from_pcd(self, xyz, rgb):
        """
        Creates Gaussians from COLMAP points.
        xyz: (N, 3) numpy array
        rgb: (N, 3) numpy array (values between 0 and 1)
        """
        # Positions
        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        
        # Colors (RGB -> 0th Degree SH coefficient conversion)
        fused_color = torch.tensor(np.asarray(rgb)).float().cuda().unsqueeze(1)
        
        # Scaling (use a fixed value for all points)
        dist2 = torch.clamp_min(torch.ones_like(fused_point_cloud[:, 0]) * 0.1, 1e-7)

        # For now derive a log-scale based on a small base distance to improve visibility.
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Rotation (Initially identity quaternion [1,0,0,0])
        # (N, 4): [w, x, y, z] with w=1 yields identity rotation
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Opacity: store as logits for sigmoid
        opacities = torch.logit(torch.clamp(0.5 * torch.ones((fused_point_cloud.shape[0], 1), device="cuda"), min=1e-4, max=0.99))

        # Save the parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(fused_color.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        print(f"Model initialized with {self._xyz.shape[0]} Gaussians.")

    def get_covariance(self):
        """
        Calculates the 3D Covariance matrix Sigma = R * S * S^T * R^T
        """
        # Scaling matrix construction
        s = self.get_scaling
        # Build a 3x3 per-gaussian diagonal matrix efficiently
        L = torch.zeros((s.shape[0], 3, 3), device=s.device)
        R = quaternion_to_rotation_matrix(self._rotation)
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        M = torch.bmm(R, L)
        # Sigma = M * M^T
        covariance = torch.bmm(M, M.transpose(1, 2))
        return covariance

    def project_gaussians_2d(self, camera):
        """
        Projects 3D Gaussians to 2D image plane.
        Σ' = J * W * Σ * W^T * J^T
        """
        xyz = self.get_xyz 
        
        # World to Camera coordinate transform
        R_mat = camera.R.to(xyz.device)
        T_vec = camera.T.to(xyz.device)
        xyz_cam = (torch.matmul(R_mat, xyz.t()) + T_vec.unsqueeze(1)).t()
        
        z = xyz_cam[:, 2:3] + 1e-7  # Prevent division by zero
        x = xyz_cam[:, 0:1]
        y = xyz_cam[:, 1:2]

        # Jacobian matrix (J) 
        f = camera.fx 
        J = torch.zeros((xyz.shape[0], 2, 3), device=xyz.device)
        J[:, 0, 0] = f / z.squeeze()
        J[:, 0, 2] = -(f * x.squeeze()) / (z.squeeze()**2)
        J[:, 1, 1] = f / z.squeeze()
        J[:, 1, 2] = -(f * y.squeeze()) / (z.squeeze()**2)

        # 3D Covariance
        cov3D = self.get_covariance() 

        # Projection: Σ_2d = J * W * Σ * W^T * J^T
        W = R_mat.unsqueeze(0).expand(xyz.shape[0], -1, -1)
        M = torch.bmm(J, W) 
        cov2d = torch.bmm(M, torch.bmm(cov3D, M.transpose(1, 2)))
        
        # Add a small diagonal bias for numerical stability
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3
        
        return cov2d, xyz_cam