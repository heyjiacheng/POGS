import torch
import numpy as np
import tyro
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional

import warp as wp
import moviepy as mpy
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pogs.tracking.optim import Optimizer
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
from nerfstudio.cameras.cameras import Cameras

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

class GaussianAnimator:
    """Animates Gaussian splats along a trajectory with smooth interpolation."""
    
    def __init__(self, optimizer: Optimizer, 
                 object_idx: int, 
                 waypoints: List[Tuple[float, float, float]], 
                 orientations: Optional[List[Tuple[float, float, float, float]]] = None,
                 duration: float = 10.0,
                 fps: int = 30):
        """Initialize the animator.
        
        Args:
            optimizer: POGS optimizer containing the scene
            object_idx: Index of the object to animate
            waypoints: List of (x, y, z) world coordinates defining the path
            orientations: Optional quaternions (w, x, y, z) for each waypoint
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


def setup_optimizer(config_path: Path) -> Optimizer:
    """Initialize the POGS optimizer."""
    # Get dataset to extract camera parameters
    from nerfstudio.utils.eval_utils import eval_setup
    _, temp_pipeline, _, _ = eval_setup(config_path)
    dataset = temp_pipeline.datamanager.train_dataset
    
    if dataset is not None and len(dataset) > 0:
        ref_camera = dataset.cameras[0]
        width, height = int(ref_camera.width.item()), int(ref_camera.height.item())
        fx, fy = ref_camera.fx.item(), ref_camera.fy.item()
        cx, cy = ref_camera.cx.item(), ref_camera.cy.item()
    else:
        width, height = 640, 480
        fx = fy = 500.0
        cx, cy = width / 2.0, height / 2.0
    
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dummy_cam_pose = torch.from_numpy(np.eye(4)[:3, :]).float()[None, :]
    
    optimizer = Optimizer(config_path, K, width, height, init_cam_pose=dummy_cam_pose)
    del temp_pipeline
    time.sleep(2)  # Wait for optimizer to load
    
    return optimizer


def find_target_object(optimizer: Optimizer, semantic_query: str) -> int:
    """Find object index matching the semantic query."""
    clip_encoder = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16", 
        clip_model_pretrained="laion2b_s34b_b88k", 
        clip_n_dims=512, 
        device='cuda:0'
    ).setup()
    
    clip_encoder.set_positives([semantic_query])
    relevancy = optimizer.get_clip_relevancy(clip_encoder)
    group_masks = optimizer.optimizer.group_masks
    
    relevancy_avg = [torch.mean(relevancy[:, 0:1][mask]) for mask in group_masks]
    target_idx = torch.argmax(torch.tensor(relevancy_avg)).item()
    
    print(f"Found object '{semantic_query}' at index {target_idx}")
    return target_idx


def create_camera_from_calibration(calibration_file: str, optimizer: Optimizer) -> Cameras:
    """Create camera from calibration file and dataset parameters."""
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
    
    # Create camera
    camera = Cameras(
        camera_to_worlds=torch.from_numpy(opengl_tf[:3, :]).float()[None, :],
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
    )
    
    # Scale to nerfstudio coordinates
    camera.camera_to_worlds[:, :3, 3] *= optimizer.dataset_scale
    print(f"Final camera position: {camera.camera_to_worlds[0, :3, 3]}")
    
    return camera


def main(
    config_path: Path = Path("outputs/box/pogs/2025-07-29_151651/config.yml"),
    semantic_query: str = "box cutter",
    waypoints_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/all_subgoals.json",
    animation_duration: float = 5.0,
    output_path: str = "gaussian_animation.mp4",
    fps: int = 30,
    show_all_objects: bool = True,
    calibration_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/src/pogs/calibration_outputs/world_to_d405.tf",
):
    """Animate Gaussian splats along a specified path using camera calibration."""
    # Initialize
    wp.init()
    print("Loading model...")
    
    # Load waypoints from JSON file
    with open(waypoints_file, 'r') as f:
        data = json.load(f)
    
    # Extract waypoints (positions only) from all subgoal_pose
    waypoints = []
    for subgoal in data['subgoals']:
        subgoal_pose = subgoal['subgoal_pose']
        # Extract only the position (first 3 elements: x, y, z)
        waypoints.append((subgoal_pose[0], subgoal_pose[1], subgoal_pose[2]))
    
    print(f"Loaded {len(waypoints)} waypoints from {waypoints_file}")
    
    # Setup optimizer and find target object
    optimizer = setup_optimizer(config_path)
    target_object_idx = find_target_object(optimizer, semantic_query)
    
    # Create camera from calibration
    camera = create_camera_from_calibration(calibration_file, optimizer)
    
    # Optionally hide other objects
    original_opacities = None
    if not show_all_objects:
        original_opacities = optimizer.pipeline.model.gauss_params["opacities"].clone()
        for idx, mask in enumerate(optimizer.group_masks_global):
            if idx != target_object_idx:
                optimizer.pipeline.model.gauss_params["opacities"][mask] = -10.0
    
    # Create animator and generate frames
    # orientations=None will use the original object's quaternion for all waypoints
    animator = GaussianAnimator(
        optimizer=optimizer,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=None,  # Use original object's quaternion
        duration=animation_duration,
        fps=fps
    )
    
    print("Generating animation...")
    frames = animator.animate(camera)
    
    # Save video
    print(f"Saving video to {output_path}...")
    out_clip = mpy.ImageSequenceClip(frames, fps=fps)
    out_clip.write_videofile(output_path, codec='libx264')
    
    # Restore original opacities
    if original_opacities is not None:
        optimizer.pipeline.model.gauss_params["opacities"] = original_opacities
    
    print(f"Animation complete! Video saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)