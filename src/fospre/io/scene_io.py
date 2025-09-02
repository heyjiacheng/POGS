"""
Scene data I/O operations for saving point clouds and metadata.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List


def save_scene_at_waypoint(waypoint_idx: int, position: np.ndarray, orientation: np.ndarray,
                          transformed_target_pcd: o3d.geometry.PointCloud,
                          static_objects_pcd: o3d.geometry.PointCloud,
                          output_base_dir: str = "outputs") -> Path:
    """Save complete scene point cloud data at a specific waypoint.
    
    Args:
        waypoint_idx: Index of the waypoint
        position: Target position [x, y, z]
        orientation: Target orientation [w, x, y, z]
        transformed_target_pcd: Transformed target object point cloud
        static_objects_pcd: Static objects point cloud
        output_base_dir: Base output directory
        
    Returns:
        Path to the created scene directory
    """
    # Create output directory structure (no timestamp)
    scene_dir = Path(output_base_dir) / "scene_animation" / f"subgoal_{waypoint_idx+1}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving scene data to: {scene_dir}")
    
    # Save individual component point clouds
    target_path = scene_dir / "moving_object.ply"
    static_path = scene_dir / "static_objects.ply"
    combined_path = scene_dir / "complete_scene.ply"
    
    # Save moving object (transformed)
    o3d.io.write_point_cloud(str(target_path), transformed_target_pcd)
    print(f"  Saved moving object ({len(transformed_target_pcd.points)} points): {target_path.name}")
    
    # Save static objects point cloud
    o3d.io.write_point_cloud(str(static_path), static_objects_pcd)
    print(f"  Saved static objects ({len(static_objects_pcd.points)} points): {static_path.name}")
    
    # Create and save combined scene
    combined_pcd = o3d.geometry.PointCloud()
    transformed_points = np.asarray(transformed_target_pcd.points)
    static_points = np.asarray(static_objects_pcd.points)
    all_points = np.vstack([transformed_points, static_points])
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Combine colors if available
    if len(transformed_target_pcd.colors) > 0 and len(static_objects_pcd.colors) > 0:
        transformed_colors = np.asarray(transformed_target_pcd.colors)
        static_colors = np.asarray(static_objects_pcd.colors)
        all_colors = np.vstack([transformed_colors, static_colors])
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    o3d.io.write_point_cloud(str(combined_path), combined_pcd)
    print(f"  Saved complete scene ({len(combined_pcd.points)} points): {combined_path.name}")
    
    return scene_dir


def save_ranked_poses(pose_rankings: List, camera, animator, output_base_dir: str, 
                     session_name: str, max_poses: int = None) -> Path:
    """Save top ranked poses with visualizations.
    
    Args:
        pose_rankings: List of (pose_data, rotation_difference) tuples
        camera: Camera for rendering
        animator: Animator instance for pose updates and rendering
        output_base_dir: Base output directory
        session_name: Name for this ranking session
        max_poses: Maximum number of poses to save
        
    Returns:
        Path to the ranked poses directory
    """
    from PIL import Image
    
    # Determine number of poses to save
    num_to_save = min(max_poses or len(pose_rankings), len(pose_rankings))
    ranked_output = f"{output_base_dir}/ranked_poses_{session_name}"
    ranked_path = Path(ranked_output)
    ranked_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving top {num_to_save} ranked poses to {ranked_output}...")
    
    for pose_idx, ranking_tuple in enumerate(pose_rankings[:num_to_save]):
        # Handle both combined scoring (4 elements) and rotation-only (2 elements)
        if len(ranking_tuple) == 4:
            pose_data, combined_score, rot_diff, trans_dist = ranking_tuple
        else:
            pose_data, rot_diff = ranking_tuple
        # Create new pose directory name with ranking
        new_pose_name = f"ranked_pose_{pose_idx+1:03d}"
        dest_dir = ranked_path / new_pose_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Update object pose and render
        animator.update_object_pose(pose_data['position'], pose_data['orientation'])
        gaussian_image = animator.render_frame(camera)
        
        # Save Gaussian render as PNG
        gaussian_path = dest_dir / "gaussian_render.png"
        gaussian_pil = Image.fromarray(gaussian_image)
        gaussian_pil.save(gaussian_path)
        
        # Save pose metadata as JSON
        pose_metadata = {
            'position': pose_data['position'].tolist(),
            'orientation': pose_data['orientation'].tolist(),
            'index': pose_data['index'],
            'rotation_difference_deg': rot_diff
        }
        if len(ranking_tuple) == 4:
            pose_metadata['combined_score'] = combined_score
            pose_metadata['translation_distance'] = trans_dist
        
        metadata_path = dest_dir / "pose_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(pose_metadata, f, indent=2)
        
        # Save point cloud files
        target_pcd_path = dest_dir / "moving_object.ply"
        static_pcd_path = dest_dir / "static_objects.ply"
        combined_pcd_path = dest_dir / "complete_scene.ply"
        
        # Save moving object point cloud
        o3d.io.write_point_cloud(str(target_pcd_path), pose_data['transformed_pcd'])
        
        # Save static objects point cloud
        o3d.io.write_point_cloud(str(static_pcd_path), animator.static_objects_pcd)
        
        # Create and save combined scene
        combined_pcd = o3d.geometry.PointCloud()
        transformed_points = np.asarray(pose_data['transformed_pcd'].points)
        static_points = np.asarray(animator.static_objects_pcd.points)
        all_points = np.vstack([transformed_points, static_points])
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        
        # Combine colors if available
        if len(pose_data['transformed_pcd'].colors) > 0 and len(animator.static_objects_pcd.colors) > 0:
            transformed_colors = np.asarray(pose_data['transformed_pcd'].colors)
            static_colors = np.asarray(animator.static_objects_pcd.colors)
            all_colors = np.vstack([transformed_colors, static_colors])
            combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        o3d.io.write_point_cloud(str(combined_pcd_path), combined_pcd)
        
        if (pose_idx + 1) % 5 == 0:
            print(f"  Saved {pose_idx + 1}/{num_to_save} poses")
    
    # Reset to original pose
    animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()

    return ranked_path