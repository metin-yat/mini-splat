import torch
import torch.nn as nn
import numpy as np
from .graphic_utils import quaternion_to_rotation_matrix

class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # These tensors are the main parameters that will be updated during training
        self._xyz = nn.Parameter(torch.empty(0))           # Positions (N, 3)
        self._features_dc = nn.Parameter(torch.empty(0))   # Colors (0th Degree SH - RGB) (N, 1, 3)
        self._scaling = nn.Parameter(torch.empty(0))       # Scales (N, 3)
        self._rotation = nn.Parameter(torch.empty(0))      # Rotations (Quaternion) (N, 4)
        self._opacity = nn.Parameter(torch.empty(0))       # Opacity (N, 1)

    # --- Activations (Physical Constraints) ---
    
    @property
    def get_scaling(self):
        # Scale cannot be negative, so we use exp()
        return torch.exp(self._scaling)

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

    # --- Initialization ---

    def create_from_pcd(self, xyz, rgb):
        """
        Creates Gaussians from COLMAP points.
        xyz: (N, 3) numpy array
        rgb: (N, 3) numpy array (values between 0 and 1)
        """
        # 1. Positions
        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        
        # 2. Colors (RGB -> 0th Degree SH coefficient conversion)
        # For simplicity, we can store RGB directly as the coefficient at the start
        fused_color = torch.tensor(np.asarray(rgb)).float().cuda().unsqueeze(1)
        
        # 3. Scaling (Initially according to distance between points)
        # For now, let's simply initialize all points with a similar average scale
        dist2 = torch.clamp_min(torch.ones_like(fused_point_cloud[:, 0]) * 0.01, 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        
        # 4. Rotation (Initially identity matrix - [1, 0, 0, 0])
        # (N, 4): [1, 0, 0, 0]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 5. Opacity (Initially a low value, e.g. 0.1)
        # We use the inverse sigmoid (logit) so that when get_opacity() is called, it yields 0.1.
        opacities = torch.logit(0.1 * torch.ones((fused_point_cloud.shape[0], 1), device="cuda"))

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
        (Sub-step 3.2 and 3.3)
        """
        # 1. Scaling matrix construction (S)
        s = self.get_scaling # (N, 3) - Exp uygulanmış hali
        
        # S matrisi diyagonal olduğu için çarpımı hızlandırmak adına 
        # direkt ölçekleri matrisin elemanlarıyla çarpan bir yapı kuruyoruz
        L = torch.zeros((s.shape[0], 3, 3), device=s.device)
        R = quaternion_to_rotation_matrix(self._rotation) # (Sub-step 3.1)
        
        # Scaling matrix S'i R ile birleştiriyoruz (M = R * S)
        # S diyagonal olduğundan her sütunu s_i ile çarpmak yeterli
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        
        M = torch.bmm(R, L) # Batch Matrix Multiplication
        
        # 2. Sigma = M * M^T (Bu RSS^TR^T'ye eşittir)
        covariance = torch.bmm(M, M.transpose(1, 2))
        
        return covariance

