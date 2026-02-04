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

def fetch_scene_data(model_path, images_path):
    # Read COLMAP data
    cameras_extrinsic, images_intrinsic, points3D = read_model(path=model_path, ext=".bin")
    
    cam_infos = []
    
    # Iterate over images and camera associations
    for idx, key in enumerate(images_intrinsic):
        colmap_image = images_intrinsic[key]
        colmap_cam = cameras_extrinsic[colmap_image.camera_id]
        
        # Rotation and Translation
        # COLMAP'ten gelen R ve T'yi standart koordinat sistemine çeviriyoruz
        # Y ve Z eksenlerini tersine çeviriyoruz
        R = np.array(torch.from_numpy(qvec2rotmat(colmap_image.qvec)).float())
        T = np.array(torch.from_numpy(colmap_image.tvec).float())
        
        R[1:3, :] *= -1  # 2. ve 3. satırı -1 ile çarp (Y ve Z ekseni)
        T[1:3] *= -1     # Translasyonun Y ve Z bileşenlerini -1 ile çarp
        
        # Şimdi tensöre çevirebiliriz
        R = torch.from_numpy(R).float()
        T = torch.from_numpy(T).float()

        # Camera parameters (PINHOLE)
        params = colmap_cam.params
        if colmap_cam.model == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif colmap_cam.model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            # Simplified assumption for other models
            fx = fy = params[0]
            cx, cy = params[1], params[2]

        img_path = os.path.join(images_path, colmap_image.name)
        
        cam_info = CameraInfo(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy, 
                              width=colmap_cam.width, height=colmap_cam.height,
                              image_path=img_path, image_name=colmap_image.name)
        cam_infos.append(cam_info)
        
    # Prepare the point cloud
    xyz = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()]) / 255.0 # Normalize (0-1)
    
    return cam_infos, xyz, rgb