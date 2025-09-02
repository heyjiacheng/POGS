#!/usr/bin/env python3
"""
Main pipeline script for combined Gaussian animation, collision filtering, and pose ranking.

This script provides a complete pipeline for:
1. Identifying target objects based on proximity to grasp stage subgoal poses
2. Generating Gaussian splat and point cloud animations
3. Sampling and filtering random poses for target objects
4. Ranking poses by rotation similarity to final waypoint
5. Interactive pose selection and new subgoals generation
6. Creating new animations with selected poses

The target object is now determined by finding the object closest to grasp stage 
subgoals (where is_grasp_stage=true in all_subgoals.json), rather than using CLIP semantic queries.
"""

import tyro
import warp as wp
import numpy as np
from pathlib import Path
from typing import Optional

# Import all our modular components
from fospre.core.utils import find_latest_config
from fospre.core.optimizer import setup_optimizer, find_target_object_by_subgoals
from fospre.core.camera import create_camera_from_calibration
from fospre.animation.animator import GaussianPointCloudAnimator
from fospre.animation.waypoints import load_waypoints_from_json, create_new_subgoals_json
from fospre.collision.detector import CollisionDetector
from fospre.pose.generator import generate_random_poses
from fospre.pose.ranking import rank_poses_by_rotation, rank_poses_by_rotation_and_translation, interactive_pose_selection
from fospre.pose.transforms import move_down_until_collision
from fospre.io.scene_io import save_scene_at_waypoint, save_ranked_poses
from fospre.io.video_io import save_animation_video, generate_new_animation
from fospre.io.multiview import generate_multiview_renders



def generate_scene_animation(animator: GaussianPointCloudAnimator, target_object_name: str, 
                            output_base_dir: str = "outputs") -> list:
    """Generate and save complete scene point cloud data at each waypoint."""
    print(f"\n=== {target_object_name.upper()} SCENE ANIMATION GENERATION ===")
    print(f"Original Gaussian position:    [{animator.original_position[0]:8.6f}, {animator.original_position[1]:8.6f}, {animator.original_position[2]:8.6f}]")
    print(f"Original PointCloud centroid:  [{animator.original_pcd_centroid[0]:8.6f}, {animator.original_pcd_centroid[1]:8.6f}, {animator.original_pcd_centroid[2]:8.6f}]")
    print(f"Target object has {len(animator.target_object_pcd.points)} point cloud points")
    print(f"Static objects have {len(animator.static_objects_pcd.points)} point cloud points")
    print(f"Generating scene data for {len(animator.waypoints)} waypoints...")
    print()
    
    saved_dirs = []
    
    for i, waypoint in enumerate(animator.waypoints):
        # Get corresponding orientation
        orientation = animator.orientations[i]
        
        # Update Gaussian pose
        animator.update_object_pose(waypoint, orientation)
        
        # Apply the transformations to the model
        animator.optimizer.optimizer.apply_to_model(
            animator.optimizer.optimizer.part_deltas,
            animator.optimizer.optimizer.group_labels
        )
        
        # Get current Gaussian position after transformation
        current_gaussian_pose = animator.optimizer.get_parts2world()[animator.object_idx]
        current_gaussian_pos = current_gaussian_pose.translation()
        
        # Transform target object point cloud
        transformed_target_pcd = animator.generate_transformed_target_pointcloud(waypoint, orientation)
        transformed_pcd_centroid = transformed_target_pcd.get_center()
        
        # Calculate deltas
        gaussian_delta = current_gaussian_pos - animator.original_position
        pcd_delta = transformed_pcd_centroid - animator.original_pcd_centroid
        
        print(f"Subgoal {i+1}/{len(animator.waypoints)}:")
        print(f"  Target waypoint:        [{waypoint[0]:8.6f}, {waypoint[1]:8.6f}, {waypoint[2]:8.6f}]")
        print(f"  Gaussian position:      [{current_gaussian_pos[0]:8.6f}, {current_gaussian_pos[1]:8.6f}, {current_gaussian_pos[2]:8.6f}]")
        print(f"  PointCloud centroid:    [{transformed_pcd_centroid[0]:8.6f}, {transformed_pcd_centroid[1]:8.6f}, {transformed_pcd_centroid[2]:8.6f}]")
        
        # Save scene data at this waypoint
        scene_dir = save_scene_at_waypoint(i, waypoint, orientation, transformed_target_pcd, 
                                         animator.static_objects_pcd, output_base_dir)
        saved_dirs.append(scene_dir)
        print()
    
    # Reset to original pose
    animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()
    
    print("=== SCENE GENERATION SUMMARY ===")
    print(f"Total waypoints:               {len(animator.waypoints)}")
    print(f"Scene directories created:     {len(saved_dirs)}")
    print(f"Final waypoint:                [{animator.waypoints[-1][0]:8.6f}, {animator.waypoints[-1][1]:8.6f}, {animator.waypoints[-1][2]:8.6f}]")
    print(f"\nScene data saved to:")
    for scene_dir in saved_dirs:
        print(f"  {scene_dir}")
    
    return saved_dirs


def process_random_poses(animator: GaussianPointCloudAnimator, camera, 
                        num_random_poses: int, max_translation: float, max_rotation_deg: float,
                        min_z: float, collision_threshold: float, max_ranked_poses: Optional[int]):
    """Process random poses: generate, filter collisions, rank, and save."""
    print(f"\n=== GENERATING RANDOM POSES ===")
    
    # Use the final waypoint as the base pose for random generation
    final_waypoint = animator.waypoints[-1]
    base_position = np.array(final_waypoint)
    base_orientation = animator.orientations[-1]  # Final orientation from waypoints
    
    print(f"Using final waypoint as base for random poses:")
    print(f"Base position: [{base_position[0]:.6f}, {base_position[1]:.6f}, {base_position[2]:.6f}]")
    print(f"Base orientation: [{base_orientation[0]:.6f}, {base_orientation[1]:.6f}, {base_orientation[2]:.6f}, {base_orientation[3]:.6f}]")
    
    # Generate random poses
    random_positions, random_orientations = generate_random_poses(
        base_position=base_position,
        base_orientation=base_orientation,
        num_poses=num_random_poses,
        max_translation=max_translation,
        min_z=min_z,
        max_rotation_deg=max_rotation_deg
    )
    
    print(f"\nProcessing random poses for collision detection and ranking...")
    
    # Initialize collision detector
    collision_detector = CollisionDetector(collision_threshold)
    
    # Process poses for collision detection
    valid_pose_data = []
    collision_count = 0
    
    for pose_idx, (pos, orient) in enumerate(zip(random_positions, random_orientations)):
        # Update object pose
        animator.update_object_pose(pos, orient)
        
        # Generate transformed point clouds for collision detection
        transformed_target_pcd = animator.generate_transformed_target_pointcloud(pos, orient)
        
        # Check collision
        collision_detected, _, _ = collision_detector.detect_collision_sdf(
            transformed_target_pcd, animator.static_objects_pcd
        )
        
        if not collision_detected:
            # Move target object down until it reaches collision threshold
            # This ensures the object is grounded and not floating
            grounded_position, collision_found = move_down_until_collision(
                pos, orient, animator, collision_detector
            )
            
            # Generate final transformed point cloud at grounded position
            final_transformed_pcd = animator.generate_transformed_target_pointcloud(
                grounded_position, orient
            )
            
            # Store valid pose data with grounded position
            pose_data = {
                'position': grounded_position,  # Use grounded position
                'orientation': orient,
                'index': pose_idx,
                'transformed_pcd': final_transformed_pcd,
                'original_position': pos,  # Keep original for reference
                'grounded': collision_found  # Track whether grounding was successful
            }
            valid_pose_data.append(pose_data)
        else:
            collision_count += 1
        
        if (pose_idx + 1) % 100 == 0:
            print(f"  Processed {pose_idx + 1}/{num_random_poses} poses "
                  f"(Valid: {len(valid_pose_data)}, Collisions: {collision_count})")
    
    # Reset to original pose
    animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()
    
    print(f"\n=== COLLISION FILTERING RESULTS ===")
    print(f"Total poses: {num_random_poses}")
    print(f"Valid poses (no collision): {len(valid_pose_data)}")
    print(f"Collision poses (filtered out): {collision_count}")
    print(f"Valid pose ratio: {len(valid_pose_data)/num_random_poses:.2%}")
    
    if not valid_pose_data:
        print("No valid poses found! Try adjusting parameters.")
        return None, None
    
    # Pose ranking
    print(f"\n=== POSE RANKING ===")
    # Use combined rotation and translation ranking
    pose_rankings = rank_poses_by_rotation_and_translation(
        valid_pose_data, base_orientation, base_position, 
        rotation_weight=0.3, translation_weight=0.7
    )
    
    # Save top ranked poses
    num_to_save = min(max_ranked_poses or len(pose_rankings), len(pose_rankings))
    session_name = "random_poses"
    
    ranked_path = save_ranked_poses(
        pose_rankings, camera, animator, "outputs", session_name, num_to_save
    )
    
    print(f"Ranking complete! Top {num_to_save} poses saved to: {ranked_path}")
    
    return pose_rankings, ranked_path


def main(
    config_path: Path = None,
    pointcloud_path: str = "outputs/box/prime_seg_gaussians.ply",
    waypoints_file: str = "outputs/all_subgoals.json",
    generate_video: bool = True,
    animation_duration: float = 5.0,
    output_video_path: str = "outputs/test_videos/gaussian_pointcloud_animation.mp4",
    fps: int = 30,
    show_all_objects: bool = True,
    calibration_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/src/pogs/calibration_outputs/world_to_d405.tf",
    enable_random_poses: bool = True,
    num_random_poses: int = 1000,
    max_translation: float = 0.08,
    max_rotation_deg: float = 90.0,
    min_z: float = 0.001,
    collision_threshold: float = 0.005,
    enable_pose_ranking: bool = True,
    max_ranked_poses: Optional[int] = 20,
    enable_multiview_renders: bool = True,
    multiview_center_point: tuple = (-0.346, -0.08, 0.02),
    multiview_num_views: int = 50,
    multiview_radius: float = 0.5,
):
    """Combined Gaussian animation, collision filtering, pose ranking, and multiview rendering pipeline.
    
    Now uses subgoal-based target object detection instead of CLIP semantic queries.
    Target object is determined by finding the object closest to grasp stage subgoal poses.
    
    After generating and ranking poses, automatically generates multiview renders 
    (50 different camera angles) around each ranked pose position.
    """
    
    # Initialize
    wp.init()
    print("Loading model...")
    
    # Find latest config if not provided
    if config_path is None:
        config_path = find_latest_config()
        print(f"Using latest config: {config_path}")
    
    # Load waypoints and setup components
    waypoints = load_waypoints_from_json(waypoints_file)
    print(f"Loaded {len(waypoints)} waypoints from {waypoints_file}")
    
    optimizer = setup_optimizer(config_path)
    target_object_idx = find_target_object_by_subgoals(optimizer, waypoints_file, pointcloud_path)
    
    # Create animator
    animator = GaussianPointCloudAnimator(
        optimizer=optimizer,
        pointcloud_path=pointcloud_path,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=None  # Use original object's quaternion
    )
    
    # Generate scene animation data
    print("\nGenerating scene animation data...")
    saved_directories = generate_scene_animation(animator, f"target_object_{target_object_idx}")
    
    # Create camera for rendering
    camera = None
    if generate_video or enable_random_poses:
        print(f"\nCreating camera from calibration...")
        camera = create_camera_from_calibration(calibration_file, optimizer)
    
    # Process random poses if enabled
    pose_rankings = None
    if enable_random_poses and enable_pose_ranking:
        pose_rankings, ranked_path = process_random_poses(
            animator, camera, num_random_poses, max_translation, max_rotation_deg,
            min_z, collision_threshold, max_ranked_poses
        )
        
        # Generate multiview renders for all ranked poses if enabled
        if pose_rankings and enable_multiview_renders:
            generate_multiview_renders(
                ranked_poses_dir=ranked_path,
                animator=animator,
                optimizer=optimizer,
                calibration_file=calibration_file,
                center_point=multiview_center_point,
                num_views=multiview_num_views,
                radius=multiview_radius,
                output_base_dir="outputs"
            )
        
        if pose_rankings:
            # Interactive pose selection
            print(f"\n=== INTERACTIVE POSE SELECTION ===")
            selected_pose_idx = interactive_pose_selection(pose_rankings)
            
            if selected_pose_idx is not None:
                # User selected a pose to replace final waypoint
                selected_pose_data = pose_rankings[selected_pose_idx - 1][0]  # Get pose_data
                # Handle both combined scoring (4 elements) and rotation-only (2 elements)
                if len(pose_rankings[selected_pose_idx - 1]) == 4:
                    # Combined scoring: (pose_data, combined_score, rotation_diff, translation_distance)
                    selected_pose_data['rotation_difference_deg'] = pose_rankings[selected_pose_idx - 1][2]
                    selected_pose_data['translation_distance_m'] = pose_rankings[selected_pose_idx - 1][3]
                    selected_pose_data['combined_score'] = pose_rankings[selected_pose_idx - 1][1]
                else:
                    # Rotation-only: (pose_data, rotation_diff)
                    selected_pose_data['rotation_difference_deg'] = pose_rankings[selected_pose_idx - 1][1]
                
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
                
                # Create new subgoals JSON
                new_subgoals_file = create_new_subgoals_json(
                    original_waypoints_file=waypoints_file,
                    selected_pose_data=selected_pose_data,
                    original_target_pose=original_target_pose,
                    original_target_orient=original_target_quat,
                    output_file="outputs/new_subgoals.json"
                )
                
                # Generate new animation
                new_video_path = generate_new_animation(
                    existing_optimizer=optimizer,
                    existing_camera=camera,
                    pointcloud_path=pointcloud_path,
                    target_object_idx=target_object_idx,
                    new_waypoints_file=new_subgoals_file,
                    original_target_pose=original_target_pose,
                    original_target_orient=original_target_quat,
                    output_video_path="outputs/test_videos/new_animation.mp4",
                    animation_duration=animation_duration,
                    fps=fps,
                    show_all_objects=show_all_objects
                )
                
                print(f"\n=== FINAL RESULTS ===")
                print(f"New subgoals file: {new_subgoals_file}")
                print(f"New animation video: {new_video_path}")
            else:
                print(f"\n=== KEEPING ORIGINAL WAYPOINTS ===")
    
    # Generate original video if requested
    if generate_video:
        print(f"\nGenerating original MP4 animation video...")
        
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
        
        # Generate original animation using generate_new_animation
        original_video_path = generate_new_animation(
            existing_optimizer=optimizer,
            existing_camera=camera,
            pointcloud_path=pointcloud_path,
            target_object_idx=target_object_idx,
            new_waypoints_file=waypoints_file,  # Use original waypoints file
            original_target_pose=original_target_pose,
            original_target_orient=original_target_quat,
            output_video_path=output_video_path,
            animation_duration=animation_duration,
            fps=fps,
            show_all_objects=show_all_objects
        )
        
        print(f"Original video saved to: {original_video_path}")
    
    print(f"\nProcessing complete!")
    print(f"Point cloud scene data saved to {len(saved_directories)} directories")
    if enable_random_poses and enable_pose_ranking:
        print(f"Pose generation, filtering, and ranking completed successfully")
    if generate_video:
        print(f"MP4 video saved to: {output_video_path}")


if __name__ == "__main__":
    tyro.cli(main)