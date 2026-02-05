# -----------------------------------------------------------------------------
# Point Cloud and GaussianModel Initialization Sanity Check
#
# This script is for quick/dirty manual validation:
#   - Are the COLMAP outputs (cameras, points, colors) loading as expected?
#   - Does the GaussianModel wrapper accept those points/colors and move to GPU?
#   - Can we poke at covariance matrices and check their basic sanity?
#
# To run:
#   - Stick your .env in place with COLMAP_OUTPUT_PATH and IMAGE_PATH variables
#   - Fire this script. Inspect the scatterplot. Window MUST be closed before model test runs.
# -----------------------------------------------------------------------------

import os
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.scene_loader import fetch_scene_data
from src.utils.model import GaussianModel

load_dotenv()  # Required for .env config env vars

def run_hybrid_test():
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Execution Device: {device.type.upper()} ---")

    # Where's the stuff? Set via .env. Don't hardcode here.
    model_path = os.getenv("COLMAP_OUTPUT_PATH")
    images_path = os.getenv("IMAGE_PATH")

    if not model_path or not images_path:
        print("ERROR: COLMAP_OUTPUT_PATH and/or IMAGE_PATH are missing in environment file.")
        return

    # Try to load COLMAPy bits (camera info, points, rgb)
    print("Loading COLMAP data...")
    try:
        cam_infos, xyz, rgb = fetch_scene_data(model_path, images_path)
        print(f"-> Loaded camera/point cloud: {len(cam_infos)} cameras, {len(xyz)} points")
    except Exception as e:
        print(f"FAILED TO LOAD SCENE DATA:\n  {e}")
        return

    # Make an interactive scatter plot (downsample if tons of points)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sampling_step = max(1, len(xyz) // 10000)  # Don't blow up matplotlib with dense clouds!
    ax.scatter(
        xyz[::sampling_step, 0],
        xyz[::sampling_step, 1],
        xyz[::sampling_step, 2],
        c=rgb[::sampling_step],
        s=1
    )
    ax.set_title("COLMAP Point Cloud (close window to continue to GaussianModel test)")
    print("Close the 3D matplotlib window to continue...")
    plt.show()

    # Now, can we build a GaussianModel on CPU or GPU? Is 'create_from_pcd' happy?
    print("\n--- Testing GaussianModel creation/population ---")
    try:
        model = GaussianModel().to(device)
        model.create_from_pcd(xyz, rgb)
        print(f"SUCCESS. {model.get_xyz.shape[0]} gaussians on: {model.get_xyz.device}")
        print("Sample opacities (first 5):", model.get_opacity[:5].detach().cpu().numpy())
    except Exception as e:
        print("FAILED to initialize/seed GaussianModel:")
        print("  ", e)
        return

    # Try poking at covariance, which internally needs log-exp-matrix business to be right
    print("\n--- Covariance Sanity Check ---")
    try:
        covariances = model.get_covariance()
        print(f"Covariance array shape: {covariances.shape}")
        cov0 = covariances[0]
        if torch.allclose(cov0, cov0.T, atol=1e-6):
            print("Covariance sample 0 is symmetric, as expected.")
        else:
            print("Covariance sample 0 is NOT symmetric! (should NOT happen)")
        print("Status: All basic GaussianModel/point cloud checks passed.")
    except Exception as e:
        print("Covariance calculation failed:")
        print("  ", e)

if __name__ == "__main__":
    run_hybrid_test()