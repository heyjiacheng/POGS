import torch
import viser
import time
import numpy as np
import tyro
from pathlib import Path
from pogs.tracking.optim import Optimizer
import warp as wp
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
import yaml
import os
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class ObjectAnimator:
    """Animates a duplicated POGS object along a specified path with smooth interpolation."""
    
    def __init__(self, server: viser.ViserServer, optimizer: Optimizer, 
                 object_idx: int, waypoints: List[Tuple[float, float, float]], 
                 orientations: Optional[List[Tuple[float, float, float, float]]] = None,
                 duration: float = 10.0):
        """
        Args:
            server: Viser server instance
            optimizer: POGS optimizer instance
            object_idx: Index of the object to animate
            waypoints: List of (x, y, z) world coordinates
            orientations: Optional list of quaternions (w, x, y, z) for each waypoint
            duration: Total animation duration in seconds
        """
        self.server = server
        self.optimizer = optimizer
        self.object_idx = object_idx
        self.waypoints = np.array(waypoints)
        self.duration = duration
        
        # Handle orientations
        if orientations is None:
            # If no orientations provided, maintain original orientation
            original_tf = self.optimizer.get_parts2world()[object_idx]
            self.orientations = [original_tf.rotation().wxyz] * len(waypoints)
        else:
            self.orientations = orientations
            
        # Get the mesh for the specified object
        self.mesh = self.optimizer.toad_object.meshes[object_idx]
        
        # Create the duplicate mesh with a unique name
        self.duplicate_name = f"animated_object_{object_idx}"
        
        # Initialize path visualization
        self.path_points = []
        self.path_name = f"{self.duplicate_name}_path"
        
        # Animation state
        self.start_time = None
        self.is_animating = True
        self.reached_end = False
        
    def hide_original_object(self):
        """Hide the original object from visualization."""
        # In viser, objects are typically managed by the POGS system
        # We don't need to explicitly hide the original since we're creating
        # a duplicate for animation purposes. The original will remain visible
        # but the duplicate will be the animated one.
        pass
            
    def create_duplicate(self):
        """Create a duplicate of the object mesh."""
        # Add the duplicate mesh to the scene
        self.server.scene.add_frame(
            self.duplicate_name,
            position=self.waypoints[0],
            wxyz=self.orientations[0],
            show_axes=True,
            axes_length=0.05,
            axes_radius=0.001
        )
        
        self.server.scene.add_mesh_trimesh(
            f"{self.duplicate_name}/mesh",
            mesh=self.mesh,
        )
        
    def interpolate_position(self, t: float) -> np.ndarray:
        """Interpolate position along waypoints using linear interpolation."""
        if t >= 1.0:
            return self.waypoints[-1]
        
        # Calculate which segment we're in
        num_segments = len(self.waypoints) - 1
        segment_duration = 1.0 / num_segments
        segment_idx = int(t / segment_duration)
        segment_t = (t - segment_idx * segment_duration) / segment_duration
        
        # Ensure we don't go out of bounds
        segment_idx = min(segment_idx, num_segments - 1)
        
        # Linear interpolation between waypoints
        start_pos = self.waypoints[segment_idx]
        end_pos = self.waypoints[segment_idx + 1]
        
        return start_pos + segment_t * (end_pos - start_pos)
    
    def interpolate_orientation(self, t: float) -> np.ndarray:
        """Interpolate orientation using SLERP."""
        if t >= 1.0:
            return self.orientations[-1]
        
        # Similar segment calculation as position
        num_segments = len(self.orientations) - 1
        segment_duration = 1.0 / num_segments
        segment_idx = int(t / segment_duration)
        segment_t = (t - segment_idx * segment_duration) / segment_duration
        
        segment_idx = min(segment_idx, num_segments - 1)
        
        # Convert to scipy Rotation objects for SLERP
        rot_start = R.from_quat([self.orientations[segment_idx][1],
                                self.orientations[segment_idx][2],
                                self.orientations[segment_idx][3],
                                self.orientations[segment_idx][0]])  # xyzw format
        rot_end = R.from_quat([self.orientations[segment_idx + 1][1],
                              self.orientations[segment_idx + 1][2],
                              self.orientations[segment_idx + 1][3],
                              self.orientations[segment_idx + 1][0]])
        
        # Create slerp interpolator
        slerp = Slerp([0, 1], R.from_quat([rot_start.as_quat(), rot_end.as_quat()]))
        interpolated = slerp(segment_t)
        
        # Convert back to wxyz format
        quat_xyzw = interpolated.as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    def update_path_visualization(self, current_pos: np.ndarray):
        """Update the path visualization with current position."""
        self.path_points.append(current_pos)
        
        if len(self.path_points) > 1:
            # Create smooth spline curve through the path points
            points = np.array(self.path_points)
            
            # Use spline for smooth path visualization
            self.server.scene.add_spline_catmull_rom(
                self.path_name,
                positions=points,
                curve_type="centripetal",
                color=(1.0, 1.0, 1.0),  # White color
                line_width=2.0,
            )
    
    def highlight_object(self):
        """Highlight the object when it reaches the end position."""
        # Create a highlighted version (golden color) by overwriting the mesh
        import copy
        highlighted_mesh = copy.deepcopy(self.mesh)
        highlighted_mesh.visual.vertex_colors = [255, 215, 0, 255]  # Gold color
        
        # Overwrite with highlighted mesh (viser will replace the existing one)
        self.server.scene.add_mesh_trimesh(
            f"{self.duplicate_name}/mesh",
            mesh=highlighted_mesh,
        )
        
        # Add a glowing frame effect
        self.server.scene.add_frame(
            f"{self.duplicate_name}_highlight",
            position=self.waypoints[-1],
            wxyz=self.orientations[-1],
            show_axes=True,
            axes_length=0.1,  # Larger axes
            axes_radius=0.005,  # Thicker axes
        )
    
    def update(self) -> bool:
        """Update animation state. Returns True if animation is complete."""
        if not self.is_animating:
            return True
            
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        t = min(elapsed / self.duration, 1.0)
        
        # Interpolate position and orientation
        current_pos = self.interpolate_position(t)
        current_orient = self.interpolate_orientation(t)
        
        # Update object transform
        self.server.scene.add_frame(
            self.duplicate_name,
            position=current_pos,
            wxyz=current_orient,
            show_axes=True,
            axes_length=0.05,
            axes_radius=0.001
        )
        
        # Update path visualization
        self.update_path_visualization(current_pos)
        
        # Check if animation is complete
        if t >= 1.0 and not self.reached_end:
            self.reached_end = True
            self.highlight_object()
            self.is_animating = False
            return True
            
        return False


def main(
    config_path: Path = Path("/home/jiachengxu/workspace/master_thesis/POGS/outputs/box/pogs/2025-07-23_204143/config.yml"),
    semantic_query: str = "blue box",
    waypoints: List[Tuple[float, float, float]] = [(0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0, 0.15), (0.3, -0.1, 0.1)],
    orientations: Optional[List[Tuple[float, float, float, float]]] = None,
    animation_duration: float = 10.0,
):
    """
    Animate a POGS object along a specified path.
    
    Args:
        config_path: Path to POGS configuration
        semantic_query: Semantic label to identify the object (e.g., "drill", "hammer")
        waypoints: List of (x, y, z) world coordinates for the path
        orientations: Optional list of quaternions (w, x, y, z) for each waypoint
        animation_duration: Total duration of the animation in seconds
    """
    # Initialize visualization server
    server = viser.ViserServer()
    wp.init()
    
    # Initialize CLIP model
    clip_encoder = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16", 
        clip_model_pretrained="laion2b_s34b_b88k", 
        clip_n_dims=512, 
        device='cuda:0'
    ).setup()
    
    # Load camera configuration (not used for animation but kept for completeness)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_filepath = os.path.join(dir_path, '../configs/camera_config.yaml')
    with open(config_filepath, 'r') as file:
        _ = yaml.safe_load(file)  # Not used in animation
    
    # Simple camera setup (we don't need actual camera feed for animation)
    camera_tf = np.eye(4)
    camera_tf[:3, 3] = [0.5, 0.5, 0.5]  # Position camera to see the scene
    
    # Initialize the optimizer with dummy camera parameters
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)  # Dummy intrinsics
    toad_opt = Optimizer(
        config_path,
        K,
        640,  # Width
        480,  # Height
        init_cam_pose=torch.from_numpy(camera_tf[:3, :]).float()[None, :],
    )
    
    # Wait for optimizer to load
    time.sleep(2)
    
    # Find the object matching the semantic query
    clip_encoder.set_positives([semantic_query])
    
    # Get relevancy scores
    relevancy = toad_opt.get_clip_relevancy(clip_encoder)
    group_masks = toad_opt.optimizer.group_masks
    
    # Find object with highest relevancy
    relevancy_avg = []
    for mask in group_masks:
        relevancy_avg.append(torch.mean(relevancy[:, 0:1][mask]))
    relevancy_avg = torch.tensor(relevancy_avg)
    target_object_idx = torch.argmax(relevancy_avg).item()
    
    print(f"Found object with semantic label '{semantic_query}' at index {target_object_idx}")
    
    # Create the animator
    animator = ObjectAnimator(
        server=server,
        optimizer=toad_opt,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=orientations,
        duration=animation_duration
    )
    
    # Hide original object and create duplicate
    animator.hide_original_object()
    animator.create_duplicate()
    
    # Animation loop
    print("Starting animation...")
    while True:
        is_complete = animator.update()
        
        if is_complete:
            print("Animation complete! Object highlighted at final position.")
            # Keep the server running to show the final state
            while True:
                time.sleep(1)
        
        time.sleep(0.03)  # ~30 FPS update rate


if __name__ == "__main__":
    tyro.cli(main)