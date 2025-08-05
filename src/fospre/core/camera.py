"""
Camera calibration and creation utilities.
"""

import torch
import numpy as np
import trimesh
from typing import Tuple

from nerfstudio.cameras.cameras import Cameras
from pogs.tracking.optim import Optimizer


def load_camera_calibration(tf_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera position and rotation from calibration file.
    
    Args:
        tf_file_path: Path to world_to_d405.tf calibration file
        
    Returns:
        Tuple of (camera_position, rotation_matrix):
        - camera_position: 3D position in world coordinates  
        - rotation_matrix: 3x3 world-to-camera rotation matrix
    """
    with open(tf_file_path, 'r') as f:
        lines = f.readlines()
    
    # Line 3: camera position in world coordinates
    camera_position = np.array([float(x) for x in lines[2].strip().split()])
    
    # Lines 4-6: rotation matrix
    rotation_matrix = np.zeros((3, 3))
    for i, line in enumerate(lines[3:6]):
        rotation_matrix[i] = [float(x) for x in line.strip().split()]
    
    return camera_position, rotation_matrix


def create_camera_from_calibration(calibration_file: str, optimizer: Optimizer) -> Cameras:
    """Create camera from calibration file and dataset parameters.
    
    Args:
        calibration_file: Path to camera calibration file
        optimizer: POGS optimizer instance
        
    Returns:
        Configured Cameras instance
    """
    # Load calibration
    camera_position, rotation_matrix = load_camera_calibration(calibration_file)
    print(f"Camera position: {camera_position}")
    
    # Create camera-to-world transform
    cam2world = np.eye(4)
    cam2world[:3, :3] = rotation_matrix.T  # Transpose for camera-to-world
    cam2world[:3, 3] = camera_position
    
    # Convert to OpenGL format for nerfstudio
    opengl_tf = cam2world @ trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    
    # Get camera intrinsics from dataset
    dataset = optimizer.pipeline.datamanager.train_dataset
    if dataset is not None and len(dataset) > 0:
        ref_camera = dataset.cameras[0]
        fx, fy = ref_camera.fx.item(), ref_camera.fy.item()
        cx, cy = ref_camera.cx.item(), ref_camera.cy.item()
        width, height = int(ref_camera.width.item()), int(ref_camera.height.item())
    else:
        fx = fy = 500.0
        cx, cy = 320.0, 240.0
        width, height = 640, 480
    
    # Create camera
    camera = Cameras(
        camera_to_worlds=torch.from_numpy(opengl_tf[:3, :]).float()[None, :],
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
    )
    
    # Scale to nerfstudio coordinates
    camera.camera_to_worlds[:, :3, 3] *= optimizer.dataset_scale
    print(f"Final camera position: {camera.camera_to_worlds[0, :3, 3]}")
    
    return camera