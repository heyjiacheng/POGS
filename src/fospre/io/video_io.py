"""
Video generation utilities for animations.
"""

import numpy as np
import moviepy as mpy
from pathlib import Path
from typing import List

from ..animation.animator import GaussianPointCloudAnimator
from ..animation.waypoints import convert_ee_waypoints_to_target_waypoints


def save_animation_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """Save animation frames as MP4 video.
    
    Args:
        frames: List of RGB frames as numpy arrays
        output_path: Output video file path
        fps: Frames per second
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving video to {output_path}...")
    out_clip = mpy.ImageSequenceClip(frames, fps=fps)
    out_clip.write_videofile(str(output_path), codec='libx264')
    print(f"Video saved successfully!")


def generate_new_animation(existing_optimizer, existing_camera, pointcloud_path: str, 
                          target_object_idx: int, new_waypoints_file: str, 
                          original_target_pose: np.ndarray, original_target_orient: np.ndarray,
                          output_video_path: str = "outputs/test_videos/new_animation.mp4",
                          animation_duration: float = 5.0, fps: int = 30,
                          show_all_objects: bool = True) -> str:
    """Generate new animation video using new subgoals and existing optimizer.
    
    Args:
        existing_optimizer: Already loaded optimizer (reuse to avoid viser reload)
        existing_camera: Already created camera
        pointcloud_path: Path to point cloud file
        target_object_idx: Target object index
        new_waypoints_file: Path to new waypoints JSON file (end effector poses)
        original_target_pose: Original target object position
        original_target_orient: Original target object orientation
        output_video_path: Output path for new animation video
        animation_duration: Animation duration in seconds
        fps: Frames per second
        show_all_objects: Whether to show all objects
        
    Returns:
        Path to generated video file
    """
    print(f"\n=== GENERATING NEW ANIMATION ===")
    print(f"Using end effector waypoints from: {new_waypoints_file}")
    print(f"Converting to target object waypoints...")
    print(f"Output video: {output_video_path}")
    
    # Convert end effector waypoints to target object waypoints
    target_waypoints, target_orientations = convert_ee_waypoints_to_target_waypoints(
        new_waypoints_file, original_target_pose, original_target_orient
    )
    print(f"Converted {len(target_waypoints)} end effector waypoints to target object waypoints")
    
    # Create animator with target object waypoints and orientations
    new_animator = GaussianPointCloudAnimator(
        optimizer=existing_optimizer,
        pointcloud_path=pointcloud_path,
        object_idx=target_object_idx,
        waypoints=target_waypoints,
        orientations=target_orientations  # Use converted target object orientations
    )
    
    # Optionally hide other objects for cleaner visualization
    original_opacities = None
    if not show_all_objects:
        original_opacities = existing_optimizer.pipeline.model.gauss_params["opacities"].clone()
        group_masks_global = existing_optimizer.optimizer.group_masks
        for idx, mask in enumerate(group_masks_global):
            if idx != target_object_idx:
                existing_optimizer.pipeline.model.gauss_params["opacities"][mask] = -10.0
    
    # Generate animation frames
    print(f"Generating animation with target object waypoints...")
    frames = new_animator.animate(existing_camera, duration=animation_duration, fps=fps)
    
    # Save video
    save_animation_video(frames, output_video_path, fps)
    
    # Restore original opacities
    if original_opacities is not None:
        existing_optimizer.pipeline.model.gauss_params["opacities"] = original_opacities
    
    print(f"New animation generation complete! Saved to {output_video_path}")
    return output_video_path