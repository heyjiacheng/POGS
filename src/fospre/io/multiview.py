"""
Multiview rendering utilities for generating multiple camera angle views.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple

from fospre.animation.animator import GaussianPointCloudAnimator
from fospre.core.camera import create_circular_cameras, load_camera_calibration
from pogs.tracking.optim import Optimizer


def generate_multiview_renders(ranked_poses_dir: str, animator: GaussianPointCloudAnimator, 
                             optimizer: Optimizer, calibration_file: str, 
                             center_point: Tuple[float, float, float] = (-0.346, -0.08, 0.02),
                             num_views: int = 50, radius: float = 0.5,
                             output_base_dir: str = "outputs"):
    """Generate multiview renders for all ranked poses.
    
    Args:
        ranked_poses_dir: Directory containing ranked poses
        animator: GaussianPointCloudAnimator instance
        optimizer: POGS optimizer instance
        calibration_file: Path to camera calibration file
        center_point: Point to look at (x, y, z)
        num_views: Number of camera views around the point
        radius: Radius of camera circle around center point
        output_base_dir: Base output directory
    """
    print(f"\n=== GENERATING MULTIVIEW RENDERS ===")
    
    ranked_path = Path(ranked_poses_dir)
    if not ranked_path.exists():
        print(f"Error: Ranked poses directory not found: {ranked_poses_dir}")
        return
    
    # Load camera calibration to get reference camera position
    ref_camera_pos, _ = load_camera_calibration(calibration_file)
    center_point_np = np.array(center_point)
    
    print(f"Reference camera position: {ref_camera_pos}")
    print(f"Center point: {center_point_np}")
    print(f"Number of views per pose: {num_views}")
    print(f"Camera radius: {radius}")
    
    # Find all ranked pose directories
    pose_dirs = sorted([d for d in ranked_path.iterdir() 
                       if d.is_dir() and d.name.startswith('ranked_pose_')])
    
    print(f"Found {len(pose_dirs)} ranked poses to process")
    
    # Process each ranked pose
    for pose_dir in pose_dirs:
        print(f"\nProcessing {pose_dir.name}...")
        
        # Load pose metadata
        metadata_path = pose_dir / "pose_metadata.json"
        if not metadata_path.exists():
            print(f"Warning: No metadata found for {pose_dir.name}, skipping")
            continue
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        pose_position = np.array(metadata['position'])
        pose_orientation = np.array(metadata['orientation'])
        
        print(f"Loaded pose: pos={pose_position}, orient={pose_orientation}")
        
        # Update animator to this pose
        try:
            animator.update_object_pose(pose_position, pose_orientation)
        except Exception as e:
            print(f"Error updating pose: {e}")
            continue
        
        # Use the pose position as the center point for camera circle
        pose_center = pose_position
        
        # Create circular cameras around the pose center  
        cameras = create_circular_cameras(
            pose_center, ref_camera_pos, num_views, radius, optimizer, calibration_file
        )
        
        # Create output directory for this pose
        output_dir = Path(output_base_dir) / "multiview_renders" / pose_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Render from all camera angles
        print(f"Rendering {num_views} views...")
        for view_idx, camera in enumerate(cameras):
            try:
                rendered_image = animator.render_frame(camera)
                
                # Save as PNG
                view_filename = f"view_{view_idx+1:03d}.png"
                view_path = output_dir / view_filename
                
                Image.fromarray(rendered_image).save(view_path)
                
                if (view_idx + 1) % 10 == 0:
                    print(f"  Rendered {view_idx + 1}/{num_views} views")
                    
            except Exception as e:
                print(f"Error rendering view {view_idx + 1}: {e}")
                continue
        
        print(f"Completed {pose_dir.name}: {num_views} views saved to {output_dir}")
    
    # Reset animator to original pose
    animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()
    print(f"Multiview rendering complete!")