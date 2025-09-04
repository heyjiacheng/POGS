"""
Camera calibration and creation utilities.
"""

import torch
import numpy as np
import trimesh
from typing import Tuple, List

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


def create_camera_from_calibration(calibration_file: str, optimizer: Optimizer, apply_centering_adjustment: bool = True) -> Cameras:
    """Create camera from calibration file and dataset parameters.
    
    Args:
        calibration_file: Path to camera calibration file
        optimizer: POGS optimizer instance
        apply_centering_adjustment: Whether to apply cx adjustment for centering objects
        
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
    
    # Conditionally adjust camera center to compensate for object appearing on right side
    if apply_centering_adjustment:
        # Shift cx to the left to center the object
        cx_adjusted = cx - width * 0.3  # Shift left by 30% of image width
    else:
        cx_adjusted = cx

    # Create camera
    camera = Cameras(
        camera_to_worlds=torch.from_numpy(opengl_tf[:3, :]).float()[None, :],
        fx=fx, fy=fy, cx=cx_adjusted, cy=cy, width=width, height=height
    )
    
    # Scale to nerfstudio coordinates
    camera.camera_to_worlds[:, :3, 3] *= optimizer.dataset_scale
    print(f"Final camera position: {camera.camera_to_worlds[0, :3, 3]}")
    print(f"Adjusted cx from {cx} to {cx_adjusted}")
    
    return camera


def create_circular_cameras(center_point: np.ndarray, ref_camera_pos: np.ndarray, 
                           num_views: int, radius: float, optimizer: Optimizer, 
                           calibration_file: str) -> List[Cameras]:
    """Create cameras positioned in a circle around the center point using successful look-at logic.
    
    Args:
        center_point: Point to look at [x, y, z]
        ref_camera_pos: Reference camera position from calibration
        num_views: Number of camera views to generate
        radius: Radius of the circle around center point
        optimizer: POGS optimizer for scaling
        calibration_file: Path to calibration file for creating reference cameras
        
    Returns:
        List of Camera objects
    """
    cameras = []
    
    # Calculate distance from reference camera to center point
    ref_distance = np.linalg.norm(ref_camera_pos - center_point)
    actual_radius = max(radius, ref_distance)
    camera_height = ref_camera_pos[2]  # Use same height as reference
    
    # Generate angles for circular positioning
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    
    for angle in angles:
        # Calculate camera position on circle
        cam_x = center_point[0] + actual_radius * np.cos(angle)
        cam_y = center_point[1] + actual_radius * np.sin(angle)
        cam_z = camera_height
        camera_pos = np.array([cam_x, cam_y, cam_z])
        
        # Create look-at direction
        look_dir = center_point - camera_pos
        look_dir = look_dir / np.linalg.norm(look_dir)
        
        # Create coordinate system
        up_world = np.array([0, 0, 1])
        right = np.cross(look_dir, up_world)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1, 0, 0])
        up = np.cross(right, look_dir)
        up = up / np.linalg.norm(up)
        
        # Build camera-to-world matrix
        cam2world = np.eye(4)
        cam2world[:3, 0] = right
        cam2world[:3, 1] = up
        cam2world[:3, 2] = -look_dir  # negative for right-handed
        cam2world[:3, 3] = camera_pos
        
        # Create camera by copying calibration camera approach (no centering adjustment for multiview)
        camera = create_camera_from_calibration(calibration_file, optimizer, apply_centering_adjustment=False)
        
        # Update the camera-to-world transform
        camera.camera_to_worlds[0] = torch.from_numpy(cam2world[:3, :]).float()
        camera.camera_to_worlds[:, :3, 3] *= optimizer.dataset_scale
        
        cameras.append(camera)
    
    return cameras