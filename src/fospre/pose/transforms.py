"""
Pose transformation utilities for end effector to target object conversion.
"""

import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def get_ee_target_relative_transform(ee_pose: np.ndarray, ee_orient: np.ndarray,
                                   target_pose: np.ndarray, target_orient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the relative transformation from end effector to target object.
    
    Args:
        ee_pose: End effector position [x, y, z]
        ee_orient: End effector orientation [w, x, y, z]
        target_pose: Target object position [x, y, z]
        target_orient: Target object orientation [w, x, y, z]
        
    Returns:
        Tuple of (relative_position, relative_orientation) in end effector frame
    """
    # Convert to rotation objects
    ee_rot = R.from_quat([ee_orient[1], ee_orient[2], ee_orient[3], ee_orient[0]])
    target_rot = R.from_quat([target_orient[1], target_orient[2], target_orient[3], target_orient[0]])
    
    # Calculate relative position in end effector frame
    position_diff = target_pose - ee_pose
    relative_position = ee_rot.inv().apply(position_diff)
    
    # Calculate relative rotation
    relative_rot = ee_rot.inv() * target_rot
    relative_quat = relative_rot.as_quat()  # [x, y, z, w]
    relative_orient = np.array([relative_quat[3], relative_quat[0], relative_quat[1], relative_quat[2]])  # [w, x, y, z]
    
    return relative_position, relative_orient


def apply_relative_transform(ee_pose: np.ndarray, ee_orient: np.ndarray,
                           relative_pos: np.ndarray, relative_orient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply relative transform to get target object pose from end effector pose.
    
    Args:
        ee_pose: End effector position [x, y, z]
        ee_orient: End effector orientation [w, x, y, z]
        relative_pos: Relative position of target in end effector frame
        relative_orient: Relative orientation of target in end effector frame [w, x, y, z]
        
    Returns:
        Tuple of (target_position, target_orientation) in world frame
    """
    # Convert to rotation objects
    ee_rot = R.from_quat([ee_orient[1], ee_orient[2], ee_orient[3], ee_orient[0]])
    relative_rot = R.from_quat([relative_orient[1], relative_orient[2], relative_orient[3], relative_orient[0]])
    
    # Transform relative position to world frame
    target_position = ee_pose + ee_rot.apply(relative_pos)
    
    # Combine rotations
    target_rot = ee_rot * relative_rot
    target_quat = target_rot.as_quat()  # [x, y, z, w]
    target_orientation = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])  # [w, x, y, z]
    
    return target_position, target_orientation


def move_down_until_collision(position: np.ndarray, orientation: np.ndarray, 
                             animator, collision_detector, step_size: float = 0.001,
                             max_steps: int = 200) -> Tuple[np.ndarray, bool]:
    """Move target object down along z-axis until collision threshold is exceeded.
    
    This function ensures the target object is not floating in air by moving it
    down until it makes sufficient contact with static objects.
    
    Args:
        position: Initial position [x, y, z]
        orientation: Object orientation [w, x, y, z]
        animator: GaussianPointCloudAnimator instance
        collision_detector: CollisionDetector instance
        step_size: Distance to move down per step (meters)
        max_steps: Maximum number of steps to prevent infinite loops
        
    Returns:
        Tuple of (final_position, collision_found)
    """
    current_position = position.copy()
    
    for step in range(max_steps):
        # Generate transformed point cloud at current position
        transformed_target_pcd = animator.generate_transformed_target_pointcloud(
            current_position, orientation
        )
        
        # Check collision
        collision_detected, collision_points, collision_ratio = collision_detector.detect_collision_sdf(
            transformed_target_pcd, animator.static_objects_pcd
        )
        
        if collision_detected:
            # Found sufficient collision - target object is grounded
            return current_position, True
        
        # Move down by step_size
        current_position[2] -= step_size
        
        # Safety check - don't go below a reasonable floor level
        if current_position[2] < -0.1:
            break
    
    # If we reach here, no sufficient collision was found
    # Return the lowest position tested
    return current_position, False