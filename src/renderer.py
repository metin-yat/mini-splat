import torch

def simple_rasterizer(model, camera, cov2d, xyz_cam):
    """
    Rasterize Gaussians to an image using alpha compositing.

    Inputs:
      model   - GaussianModel instance (provides colors, opacity)
      camera  - CameraInfo with intrinsics/extrinsics
      cov2d   - per-gaussian 2x2 covariances in image plane
      xyz_cam - gaussian centers in camera coordinates (N, 3)
    """
    h, w = camera.height, camera.width
    f = camera.fx
    device = xyz_cam.device

    # Convert 3D camera-space centers to 2D pixel coordinates
    means2D_x = (xyz_cam[:, 0] * f / xyz_cam[:, 2]) + camera.cx
    means2D_y = (xyz_cam[:, 1] * f / xyz_cam[:, 2]) + camera.cy
    means2D = torch.stack([means2D_x, means2D_y], dim=-1)

    # Depth sorting: back-to-front for alpha compositing
    indices = torch.argsort(xyz_cam[:, 2], descending=True)

    means2D = means2D[indices]
    cov2d = cov2d[indices]
    # Colors: (N, 1, 3) -> (N, 3)
    colors = model.get_features()[indices].squeeze(1)
    opacities = model.get_opacity[indices]

    # Pixel grid (H, W, 2)
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device),
                                    indexing='ij')
    pixel_coords = torch.stack([grid_x, grid_y], dim=-1).float()

    canvas = torch.zeros((h, w, 3), device=device)
    T = torch.ones((h, w, 1), device=device)  # Transmittance

    # Alpha compositing over a limited set of gaussians for memory reasons
    for i in range(len(indices)):
        d = pixel_coords - means2D[i]
        inv_cov = torch.inverse(cov2d[i])

        # Gaussian density exponent (2D)
        power = -0.5 * (d[:, :, 0]**2 * inv_cov[0, 0] +
                        d[:, :, 1]**2 * inv_cov[1, 1] +
                        2 * d[:, :, 0] * d[:, :, 1] * inv_cov[0, 1])

        # Per-pixel alpha (clamped)
        alpha = torch.clamp(torch.exp(power).unsqueeze(-1) * opacities[i], max=0.99)

        # Composite color and update transmittance
        canvas = canvas + colors[i] * alpha * T
        T = T * (1 - alpha)

    return canvas