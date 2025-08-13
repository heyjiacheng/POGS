#!/usr/bin/env python3
"""
Generate animation using action_subgoals.json directly.

This script uses the generate_new_animation function to create Gaussian splat
animations based on action subgoals that contain end effector poses and 
grasp/release stage information.
"""

import tyro
import warp as wp
import numpy as np
from pathlib import Path

# Import all our modular components
from fospre.core.utils import find_latest_config
from fospre.core.optimizer import setup_optimizer, find_target_object_by_subgoals
from fospre.core.camera import create_camera_from_calibration
from fospre.animation.animator import GaussianPointCloudAnimator
from fospre.animation.waypoints import load_waypoints_from_json
from fospre.io.video_io import generate_new_animation


def main(
    config_path: Path = None,
    pointcloud_path: str = "outputs/box/prime_seg_gaussians.ply",
    action_subgoals_file: str = "outputs/action_subgoals.json",
    animation_duration: float = 5.0,
    output_video_path: str = "outputs/test_videos/action_animation.mp4",
    fps: int = 30,
    show_all_objects: bool = True,
    calibration_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/src/pogs/calibration_outputs/world_to_d405.tf",
):
    """Generate animation using action_subgoals.json file.
    
    This script uses generate_new_animation function to create animations based on
    action subgoals that include grasp/release stage information and end effector poses.
    """
    
    # Initialize
    wp.init()
    print("Loading model...")
    
    # Find latest config if not provided
    if config_path is None:
        config_path = find_latest_config()
        print(f"Using latest config: {config_path}")
    
    # Load waypoints from action_subgoals.json (for target object detection)
    waypoints = load_waypoints_from_json(action_subgoals_file)
    print(f"Loaded {len(waypoints)} waypoints from {action_subgoals_file}")
    
    optimizer = setup_optimizer(config_path)
    target_object_idx = find_target_object_by_subgoals(optimizer, action_subgoals_file, pointcloud_path)
    
    # Create animator to get original target object pose and orientation
    animator = GaussianPointCloudAnimator(
        optimizer=optimizer,
        pointcloud_path=pointcloud_path,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=None  # Use original object's quaternion
    )
    
    # Create camera for rendering
    print(f"\nCreating camera from calibration...")
    camera = create_camera_from_calibration(calibration_file, optimizer)
    
    # Get original target object pose for calculations
    original_target_pose = animator.original_position
    if hasattr(animator.original_position, 'cpu'):
        original_target_pose = animator.original_position.cpu().numpy()
    
    original_target_rotation = animator.original_rotation
    if hasattr(original_target_rotation, 'wxyz'):
        original_target_quat = original_target_rotation.wxyz
        if hasattr(original_target_quat, 'cpu'):
            original_target_quat = original_target_quat.cpu().numpy()
    else:
        # Fallback to identity quaternion
        original_target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    
    # Generate animation using action_subgoals.json directly
    print(f"\nGenerating animation from action subgoals...")
    new_video_path = generate_new_animation(
        existing_optimizer=optimizer,
        existing_camera=camera,
        pointcloud_path=pointcloud_path,
        target_object_idx=target_object_idx,
        new_waypoints_file=action_subgoals_file,  # Use action_subgoals.json directly
        original_target_pose=original_target_pose,
        original_target_orient=original_target_quat,
        output_video_path=output_video_path,
        animation_duration=animation_duration,
        fps=fps,
        show_all_objects=show_all_objects
    )
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Action subgoals file: {action_subgoals_file}")
    print(f"Animation video: {new_video_path}")


if __name__ == "__main__":
    tyro.cli(main)