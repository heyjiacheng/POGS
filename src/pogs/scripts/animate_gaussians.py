import torch
import numpy as np
import tyro
from pathlib import Path
from pogs.tracking.optim import Optimizer
import warp as wp
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import viser.transforms as vtf
import moviepy as mpy
from nerfstudio.models.splatfacto import SH2RGB
import time
from copy import deepcopy
from nerfstudio.cameras.cameras import Cameras
import trimesh

class GaussianAnimator:
    """Animates Gaussian splats along a specified trajectory."""
    
    def __init__(self, optimizer: Optimizer, 
                 object_idx: int, 
                 waypoints: List[Tuple[float, float, float]], 
                 orientations: Optional[List[Tuple[float, float, float, float]]] = None,
                 duration: float = 10.0,
                 fps: int = 30):
        """
        Args:
            optimizer: POGS optimizer instance
            object_idx: Index of the object to animate
            waypoints: List of (x, y, z) world coordinates
            orientations: Optional list of quaternions (w, x, y, z) for each waypoint
            duration: Total animation duration in seconds
            fps: Frames per second for the output video
        """
        self.optimizer = optimizer
        self.object_idx = object_idx
        self.waypoints = np.array(waypoints)
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        
        # Handle orientations
        if orientations is None:
            # If no orientations provided, maintain original orientation
            original_tf = self.optimizer.get_parts2world()[object_idx]
            self.orientations = [original_tf.rotation().wxyz] * len(waypoints)
        else:
            self.orientations = orientations
            
        # Get the original object transformation
        self.original_pose = self.optimizer.get_parts2world()[object_idx]
        self.original_position = self.original_pose.translation()
        self.original_rotation = self.original_pose.rotation()
        
        # Store original part deltas
        self.original_part_deltas = self.optimizer.optimizer.part_deltas.clone()
        
        # Get initial part to world transform for proper delta calculation
        self.initial_part2world = self.optimizer.optimizer.get_initial_part2world(object_idx)
        
    def interpolate_position(self, t: float) -> np.ndarray:
        """Interpolate position along waypoints using linear interpolation."""
        if t >= 1.0:
            return self.waypoints[-1]
        
        # Calculate which segment we're in
        num_segments = len(self.waypoints) - 1
        if num_segments == 0:
            return self.waypoints[0]
            
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
        if num_segments == 0:
            return self.orientations[0]
            
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
    
    def update_object_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Update the pose of the specified object in the optimizer."""
        # Get initial position in world frame
        initial_position = self.initial_part2world[:3, 3].cpu().numpy()
        
        # Calculate the delta from initial position to target position
        position_delta = position - initial_position
        
        # Convert orientation to rotation matrix
        rot_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        target_rotation = R.from_quat(rot_xyzw)
        
        # Since initial rotation is identity, relative rotation equals target rotation
        relative_quat = target_rotation.as_quat()  # xyzw format
        
        # Update part_deltas for the specific object
        # Part deltas format: [x, y, z, qw, qx, qy, qz]
        new_delta = torch.tensor([
            position_delta[0],
            position_delta[1],
            position_delta[2],
            relative_quat[3],  # w
            relative_quat[0],  # x
            relative_quat[1],  # y
            relative_quat[2],  # z
        ], dtype=torch.float32, device=self.optimizer.optimizer.part_deltas.device)
        
        # Update only the specific object's delta
        self.optimizer.optimizer.part_deltas = self.original_part_deltas.clone()
        self.optimizer.optimizer.part_deltas[self.object_idx] = new_delta
    
    def render_frame(self, camera):
        """Render a single frame with the current object pose."""
        # Apply the current transformations to the model
        self.optimizer.optimizer.apply_to_model(
            self.optimizer.optimizer.part_deltas,
            self.optimizer.optimizer.group_labels
        )
        
        # Render the scene
        outputs = self.optimizer.pipeline.model.get_outputs(
            camera.to('cuda'),
            tracking=False,
            BLOCK_WIDTH=16,
            rgb_only=True
        )
        
        # Extract RGB image
        rgb_image = outputs["rgb"].squeeze().detach().cpu().numpy()
        return (rgb_image * 255).astype(np.uint8)
    
    def animate(self, camera) -> List[np.ndarray]:
        """Generate animation frames."""
        frames = []
        
        print(f"Generating {self.total_frames} frames...")
        
        for frame_idx in range(self.total_frames):
            t = frame_idx / (self.total_frames - 1) if self.total_frames > 1 else 0
            
            # Interpolate position and orientation
            current_pos = self.interpolate_position(t)
            current_orient = self.interpolate_orientation(t)
            
            # Update object pose
            self.update_object_pose(current_pos, current_orient)
            
            # Render frame
            frame = self.render_frame(camera)
            frames.append(frame)
            
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{self.total_frames}")
        
        # Reset to original pose
        self.optimizer.optimizer.part_deltas = self.original_part_deltas.clone()
        
        return frames


def main(
    config_path: Path = Path("/home/jiachengxu/workspace/master_thesis/POGS/outputs/box/pogs/2025-07-23_204143/config.yml"),
    semantic_query: str = "blue box",
    waypoints: List[Tuple[float, float, float]] = [(0, 0, 0), (0.1, 0.1, 0.1), (0.2, 0, 0.15), (0.3, -0.1, 0.1)],
    orientations: Optional[List[Tuple[float, float, float, float]]] = None,
    animation_duration: float = 5.0,
    output_path: str = "gaussian_animation.mp4",
    fps: int = 30,
    show_all_objects: bool = True,
    camera_distance: float = 1.0,
    camera_height: float = 0.5,
    camera_angle: float = 30.0,
):
    """
    Animate Gaussian splats along a specified path.
    
    Args:
        config_path: Path to POGS configuration
        semantic_query: Semantic label to identify the object (e.g., "drill", "hammer")
        waypoints: List of (x, y, z) world coordinates for the path
        orientations: Optional list of quaternions (w, x, y, z) for each waypoint
        animation_duration: Total duration of the animation in seconds
        output_path: Path for the output video file
        fps: Frames per second for the output video
        show_all_objects: Whether to show all objects or only the animated one
        camera_distance: Distance of camera from scene center
        camera_height: Height of camera above scene
        camera_angle: Vertical angle of camera in degrees
    """
    # Initialize warp
    wp.init()
    
    # Initialize CLIP model
    clip_encoder = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16", 
        clip_model_pretrained="laion2b_s34b_b88k", 
        clip_n_dims=512, 
        device='cuda:0'
    ).setup()
    
    # Load the optimizer first to get scene information
    print("Loading model...")
    
    # Use a dummy camera just for initialization (matching track_main_online_demo.py approach)
    dummy_camera_tf = np.eye(4)
    
    # Get dataset first to get proper camera dimensions
    from nerfstudio.utils.eval_utils import eval_setup
    _, temp_pipeline, _, _ = eval_setup(config_path)
    dataset = temp_pipeline.datamanager.train_dataset
    
    if dataset is not None and len(dataset) > 0:
        # Use the first camera's properties as reference for initialization
        ref_camera = dataset.cameras[0]
        init_width = int(ref_camera.width.item())
        init_height = int(ref_camera.height.item())
        init_fx = ref_camera.fx.item()
        init_fy = ref_camera.fy.item()
        init_cx = ref_camera.cx.item()
        init_cy = ref_camera.cy.item()
    else:
        # Fallback to reasonable defaults
        init_width = 640
        init_height = 480
        init_fx = init_fy = 500.0
        init_cx = init_width / 2.0
        init_cy = init_height / 2.0
    
    dummy_K = np.array([[init_fx, 0.0, init_cx], [0.0, init_fy, init_cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    
    toad_opt = Optimizer(
        config_path,
        dummy_K,
        init_width,
        init_height,
        init_cam_pose=torch.from_numpy(dummy_camera_tf[:3, :]).float()[None, :],
    )
    
    del temp_pipeline  # Clean up temporary pipeline
    
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
    
    # Get the scene bounds from waypoints to position camera properly
    waypoints_array = np.array(waypoints)
    
    # Get initial object positions from the optimizer to better center the camera
    parts2world = toad_opt.get_parts2world()
    object_positions = np.array([tf.translation() for tf in parts2world])
    
    # Use the actual object center as the look-at point
    look_at = np.mean(object_positions, axis=0)
    
    # Position camera at a reasonable distance from the objects
    max_extent = np.max(np.linalg.norm(object_positions - look_at, axis=1))
    camera_distance = max(camera_distance, max_extent * 2.5)  # Ensure camera is far enough
    
    # Calculate camera position based on the object center
    camera_angle_rad = np.radians(camera_angle)
    camera_x = look_at[0] + camera_distance * np.cos(camera_angle_rad)
    camera_y = look_at[1] - camera_distance * 0.3  # Slight offset for better viewing angle
    camera_z = look_at[2] + camera_height + camera_distance * np.sin(camera_angle_rad)
    
    camera_position = np.array([camera_x, camera_y, camera_z])
    up = np.array([0, 0, 1])
    
    # Calculate camera rotation matrix (look-at matrix)
    forward = look_at - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # Build camera transformation matrix (world to camera, OpenCV convention)
    camera_tf = np.eye(4)
    camera_tf[:3, 0] = right
    camera_tf[:3, 1] = -up  # OpenCV has Y pointing down
    camera_tf[:3, 2] = -forward  # OpenCV has Z pointing away from scene
    camera_tf[:3, 3] = camera_position
    
    # Convert to camera-to-world matrix (inverse of world-to-camera)
    cam2world_tf = np.linalg.inv(camera_tf)
    
    # Convert to OpenGL format for nerfstudio (same as in track_main_online_demo.py)
    opengl_tf = cam2world_tf @ trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    
    # Use the proper camera intrinsics from the dataset
    dataset = toad_opt.pipeline.datamanager.train_dataset
    if dataset is not None and len(dataset) > 0:
        # Use the first camera's intrinsics as reference
        ref_camera = dataset.cameras[0]
        fx = ref_camera.fx.item()
        fy = ref_camera.fy.item()
        cx = ref_camera.cx.item()
        cy = ref_camera.cy.item()
        width = int(ref_camera.width.item())
        height = int(ref_camera.height.item())
    else:
        # Use the initialization values as fallback
        fx = init_fx
        fy = init_fy
        cx = init_cx
        cy = init_cy
        width = init_width
        height = init_height
    
    # Create proper camera for rendering (similar to track_main_online_demo.py)
    render_camera = Cameras(
        camera_to_worlds=torch.from_numpy(opengl_tf[:3, :]).float()[None, :],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=int(width),
        height=int(height),
    )
    
    # Scale camera transform to nerfstudio coordinates (matching dataset scale)
    render_camera.camera_to_worlds[:, :3, 3] *= toad_opt.dataset_scale
    
    print(f"Camera position: {camera_position}")
    print(f"Looking at: {look_at}")
    print(f"Object center: {look_at}")
    print(f"Scene bounds: {waypoints_array.min(axis=0)} to {waypoints_array.max(axis=0)}")
    print(f"Camera distance: {camera_distance}")
    print(f"Dataset scale: {toad_opt.dataset_scale}")
    print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # If not showing all objects, mask out others
    if not show_all_objects:
        # Store original opacities
        original_opacities = toad_opt.pipeline.model.gauss_params["opacities"].clone()
        
        # Set opacities of other objects to very low
        for idx, mask in enumerate(toad_opt.group_masks_global):
            if idx != target_object_idx:
                toad_opt.pipeline.model.gauss_params["opacities"][mask] = -10.0
    
    # Create the animator
    animator = GaussianAnimator(
        optimizer=toad_opt,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=orientations,
        duration=animation_duration,
        fps=fps
    )
    
    # Generate animation frames
    print("Generating animation...")
    frames = animator.animate(render_camera)
    
    # Save video
    print(f"Saving video to {output_path}...")
    out_clip = mpy.ImageSequenceClip(frames, fps=fps)
    out_clip.write_videofile(output_path, codec='libx264')
    
    # Restore original opacities if modified
    if not show_all_objects:
        toad_opt.pipeline.model.gauss_params["opacities"] = original_opacities
    
    print(f"Animation complete! Video saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)