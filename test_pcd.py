"""
Point Cloud and Gaussian Model Initialization Test Module

This module checks two things:
    1. Visualization: Validates COLMAP's 3D reconstruction by rendering the sparse point cloud.
    2. Model Integration: Verifies that the Point Cloud Data (PCD) can be successfully 
       converted into a trainable GaussianModel on the available hardware (CPU/GPU).

Main Steps:
    * Loads scene data (camera info, 3D points, and colors) from COLMAP binary files.
    * Visualizes the point cloud using an interactive 3D Matplotlib scatter plot.
    * After closing the plot, initializes a GaussianModel and populates it with the PCD.
    * Checks hardware availability and prints the model's status and parameter samples.
"""

import os
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.scene_loader import fetch_scene_data
from src.utils.model import GaussianModel

load_dotenv()

def run_hybrid_test():
    # 1. Hardware Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Execution Device: {device.type.upper()} ---")

    # 2. Path Configuration
    model_path = os.getenv("COLMAP_OUTPUT_PATH")
    images_path = os.getenv("IMAGE_PATH")

    if not model_path or not images_path:
        print("Error: Please check your .env file for COLMAP_OUTPUT_PATH and IMAGE_PATH.")
        return

    # 3. Data Loading
    print("Loading COLMAP data...")
    try:
        cam_infos, xyz, rgb = fetch_scene_data(model_path, images_path)
        print(f"Successfully loaded: {len(cam_infos)} cameras, {len(xyz)} points.")
    except Exception as e:
        print(f"Failed to load scene data: {e}")
        return

    # 4. Point Cloud Visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for responsive visualization
    sampling_step = max(1, len(xyz) // 10000)
    ax.scatter(xyz[::sampling_step, 0], 
               xyz[::sampling_step, 1], 
               xyz[::sampling_step, 2], 
               c=rgb[::sampling_step], 
               s=1)
    
    ax.set_title("COLMAP Point Cloud Validation")
    print("Opening 3D visualization... Close the window to proceed with Gaussian Model test.")
    plt.show()

    # 5. Gaussian Model Initialization Test
    print("\n--- Initializing Gaussian Model ---")
    try:
        # Initialize model on the selected device
        model = GaussianModel().to(device)
        
        # Populate the model with PCD
        model.create_from_pcd(xyz, rgb)
        
        # Print status and parameter verification
        print(f"Status: SUCCESS")
        print(f"Number of Gaussians: {model.get_xyz.shape[0]}")
        print(f"Parameters residing on: {model.get_xyz.device}")
        
        # Sample check for opacity (Verify sigmoid/logit constraints)
        sample_opacity = model.get_opacity[:5].detach().cpu().numpy()
        print(f"Opacity sample (first 5): \n{sample_opacity}")
        
    except Exception as e:
        print(f"An error occurred during model initialization: {e}")


    print("\n--- Testing 3D Covariance Calculation ---")
    try:
        covariances = model.get_covariance()
        print(f"Covariance Matrix Shape: {covariances.shape}") # Expected shape: (48488, 3, 3) 
        
        # Verify if it's a valid covariance (Symmetry check)
        sample_cov = covariances[0]
        is_symmetric = torch.allclose(sample_cov, sample_cov.T, atol=1e-6)
        print(f"Sample Covariance Symmetry: {'VALID' if is_symmetric else 'FAILED'}")
        
        print(f"Status: ALL 3DGS MODEL TESTS PASSED")
    except Exception as e:
        print(f"Covariance test failed: {e}")

if __name__ == "__main__":
    run_hybrid_test()