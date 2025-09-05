"""
Random pose generation utilities.
"""

import numpy as np
import random
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R


def generate_random_poses(base_position: np.ndarray, base_orientation: np.ndarray, 
                         num_poses: int = 60, max_translation: float = 0.05, 
                         min_z: float = 0.005, max_rotation_deg: float = 180.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate random poses around a base pose with constraints.
    
    Args:
        base_position: Base position [x, y, z]
        base_orientation: Base orientation quaternion [w, x, y, z]
        num_poses: Number of random poses to generate
        max_translation: Maximum translation in each axis (meters)
        min_z: Minimum z-coordinate (above table surface)
        max_rotation_deg: Maximum rotation in each axis (degrees)
    
    Returns:
        Tuple of (positions, orientations) lists
    """
    print(f"Generating {num_poses} random poses...")
    print(f"Base position: [{base_position[0]:.6f}, {base_position[1]:.6f}, {base_position[2]:.6f}]")
    print(f"Translation constraints: ±{max_translation}m per axis, z ≥ {min_z}m")
    print(f"Rotation constraints: ±{max_rotation_deg}° per axis")
    
    positions = []
    orientations = []
    
    # Convert base orientation to rotation matrix
    base_rot_xyzw = np.array([base_orientation[1], base_orientation[2], base_orientation[3], base_orientation[0]])
    base_rotation = R.from_quat(base_rot_xyzw)
    
    # Set random seed for reproducibility
    np.random.seed(2)
    random.seed(2)

    for i in range(num_poses):
        # Generate random translation
        translation_offset = np.random.uniform(-max_translation, max_translation, 3)
        new_position = base_position + translation_offset
        
        # Ensure z constraint
        new_position[2] = max(new_position[2], min_z)
        
        # Generate random rotation
        max_rotation_rad = np.radians(max_rotation_deg)
        rotation_angles = np.random.uniform(-max_rotation_rad, max_rotation_rad, 3)
        
        # Create rotation from Euler angles
        random_rotation = R.from_euler('xyz', rotation_angles)
        
        # Combine with base rotation
        final_rotation = base_rotation * random_rotation
        
        # Convert back to wxyz quaternion format
        quat_xyzw = final_rotation.as_quat()
        final_orientation = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        positions.append(new_position)
        orientations.append(final_orientation)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_poses} poses")
    
    print(f"Successfully generated {len(positions)} random poses")
    return positions, orientations