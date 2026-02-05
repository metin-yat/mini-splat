import torch
import torch.nn.functional as F
from src.utils.model import GaussianModel
from src.renderer import simple_rasterizer
from src.scene_loader import fetch_scene_data
from src.utils.config import Config
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def ssim(img1, img2, window_size=11):
    """Simple SSIM approximation for training loss."""
    mu_x = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    sigma_x = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu_x**2
    sigma_y = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu_y**2
    sigma_xy = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu_x * mu_y

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()

def train():
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data and model setup
    # Keep max_points and target_res conservative to avoid heavy hardware use
    cam_infos, xyz, rgb = fetch_scene_data(
        Config.SPARSE_PATH,
        Config.IMAGES_PATH,
        target_res=Config.RESOLUTION,
        max_points=500
    )

    model = GaussianModel().cuda()
    model.create_from_pcd(xyz, rgb)

    optimizer = torch.optim.Adam([
        {'params': [model._xyz], 'lr': 0.00016},
        {'params': [model._features_dc], 'lr': 0.0025},
        {'params': [model._opacity], 'lr': 0.05},
        {'params': [model._scaling], 'lr': 0.005},
        {'params': [model._rotation], 'lr': 0.001},
    ])

    print("Starting training...")
    print(f"Planned iterations: {Config.ITERATIONS}")
    progress_bar = tqdm(range(Config.ITERATIONS), desc="training")
    for iteration in progress_bar:
        # Randomly sample a camera each step
        idx = np.random.randint(0, len(cam_infos))
        viewpoint_cam = cam_infos[idx]

        # Load and prepare ground-truth image
        gt_image = Image.open(viewpoint_cam.image_path).convert("RGB")
        gt_image = gt_image.resize((viewpoint_cam.width, viewpoint_cam.height))
        gt_tensor = torch.from_numpy(np.array(gt_image)).float().cuda() / 255.0

        # 2. Forward: projection then rasterization
        cov2d, xyz_cam = model.project_gaussians_2d(viewpoint_cam)
        image = simple_rasterizer(model, viewpoint_cam, cov2d, xyz_cam)

        # 3. Loss (L1 + SSIM)
        l1_loss = F.l1_loss(image, gt_tensor)

        # SSIM expects shape (1, C, H, W)
        img1 = image.permute(2, 0, 1).unsqueeze(0)
        img2 = gt_tensor.permute(2, 0, 1).unsqueeze(0)
        ssim_value = ssim(img1, img2)

        # Total loss: 0.8 * L1 + 0.2 * (1 - SSIM)
        loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_value)

        # 4. Backward and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "L1": f"{l1_loss.item():.4f}",
            "Pts": model._xyz.shape[0]
        })

        # Save a sample every 100 iterations
        if iteration % 100 == 0:
            out_img = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            save_path = os.path.join(output_dir, f"render_{iteration}.png")
            Image.fromarray(out_img).save(save_path)

if __name__ == "__main__":
    train()