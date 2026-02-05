import os
import torch
import numpy as np
from PIL import Image
from src.utils.read_write_model import read_model, qvec2rotmat

class CameraInfo:
    def __init__(self, R, T, fx, fy, cx, cy, width, height, image_path, image_name):
        self.R = R
        self.T = T
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.image_path = image_path
        self.image_name = image_name

def fetch_scene_data(model_path, images_path, target_res=400, max_points=1000):
    """
    Load COLMAP cameras and points and return camera infos and subsampled point cloud.

    Args:
      model_path: path to COLMAP model folder
      images_path: path to images folder
      target_res: target pixel size for the image long edge
      max_points: maximum number of points to load (hardware safety)
    """
    # Read COLMAP data
    cameras_extrinsic, images_intrinsic, points3D = read_model(path=model_path, ext=".bin")

    cam_infos = []

    # Iterate over images and build CameraInfo objects
    for idx, key in enumerate(images_intrinsic):
        colmap_image = images_intrinsic[key]
        colmap_cam = cameras_extrinsic[colmap_image.camera_id]

        # 1. Original dimensions and scale
        orig_w = colmap_cam.width
        orig_h = colmap_cam.height
        scale = target_res / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 2. Rotation and translation (apply coordinate adjustments)
        R = np.array(torch.from_numpy(qvec2rotmat(colmap_image.qvec)).float())
        T = np.array(torch.from_numpy(colmap_image.tvec).float())
        R[1:3, :] *= -1
        T[1:3] *= -1

        R = torch.from_numpy(R).float()
        T = torch.from_numpy(T).float()

        # 3. Scale intrinsics for the resized image
        params = colmap_cam.params
        if colmap_cam.model == "PINHOLE":
            fx, fy, cx, cy = params[0] * scale, params[1] * scale, params[2] * scale, params[3] * scale
        elif colmap_cam.model == "SIMPLE_PINHOLE":
            fx = fy = params[0] * scale
            cx, cy = params[1] * scale, params[2] * scale
        else:
            # Fallback scaling for other models
            fx = fy = params[0] * scale
            cx, cy = params[1] * scale, params[2] * scale

        img_path = os.path.join(images_path, colmap_image.name)

        # CameraInfo holds resized intrinsics for later projection
        cam_info = CameraInfo(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy,
                              width=new_w, height=new_h,
                              image_path=img_path, image_name=colmap_image.name)
        cam_infos.append(cam_info)

    # 4. Prepare the point cloud with subsampling
    xyz = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()]) / 255.0

    # If there are too many points, randomly subsample for safety
    if len(xyz) > max_points:
        print(f"Too many points ({len(xyz)}). Subsampling to {max_points} points...")
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]

    return cam_infos, xyz, rgb