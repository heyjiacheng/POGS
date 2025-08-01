import torch
import numpy as np
import tyro
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional
import random

import warp as wp
import moviepy as mpy
import trimesh
import open3d as o3d
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


class GaussianPointCloudAnimator:
    """Animates both Gaussian splats and corresponding point clouds for a target object."""
    
    def __init__(self, optimizer: Optimizer, 
                 pointcloud_path: str,
                 object_idx: int, 
                 waypoints: List[Tuple[float, float, float]], 
                 orientations: Optional[List[Tuple[float, float, float, float]]] = None):
        """Initialize the animator.
        
        Args:
            optimizer: POGS optimizer containing the scene
            pointcloud_path: Path to the PLY point cloud file
            object_idx: Index of the object to animate
            waypoints: List of (x, y, z) world coordinates defining the path
            orientations: Optional quaternions (w, x, y, z) for each waypoint
        """
        self.optimizer = optimizer
        self.pointcloud_path = pointcloud_path
        self.object_idx = object_idx
        self.waypoints = np.array(waypoints)
        
        # Load full point cloud
        self.full_pcd = o3d.io.read_point_cloud(pointcloud_path)
        if len(self.full_pcd.points) == 0:
            raise ValueError(f"Failed to load point cloud from {pointcloud_path}")
        
        print(f"Loaded full point cloud with {len(self.full_pcd.points)} points")
        
        # Handle orientations
        if orientations is None:
            # If no orientations provided, maintain original orientation
            original_tf = self.optimizer.get_parts2world()[object_idx]
            self.orientations = [original_tf.rotation().wxyz] * len(waypoints)
        else:
            self.orientations = orientations
            
        # Get the original object transformation from Gaussians
        self.original_pose = self.optimizer.get_parts2world()[object_idx]
        self.original_position = self.original_pose.translation()
        self.original_rotation = self.original_pose.rotation()
        
        print(f"Original Gaussian object position: {self.original_position}")
        
        # Store original part deltas
        self.original_part_deltas = self.optimizer.optimizer.part_deltas.clone()
        
        # Get initial part to world transform for proper delta calculation
        self.initial_part2world = self.optimizer.optimizer.get_initial_part2world(object_idx)
        
        # Extract target object point cloud using Gaussian masks
        self.target_object_pcd = self._extract_target_object_pointcloud()
        
        # Extract static objects point cloud (everything except target object)
        self.static_objects_pcd = self._extract_static_objects_pointcloud()
        
    def _extract_target_object_pointcloud(self) -> o3d.geometry.PointCloud:
        """Extract point cloud points that correspond to the target object using Gaussian masks."""
        # Get the Gaussian mask for target object
        group_masks = self.optimizer.optimizer.group_masks
        if self.object_idx >= len(group_masks):
            print(f"Warning: Object index {self.object_idx} out of range, using full point cloud")
            return self.full_pcd
        
        target_mask = group_masks[self.object_idx]
        
        # Get Gaussian positions for the target object
        gaussian_positions = self.optimizer.pipeline.model.means[target_mask].detach().cpu().numpy()
        
        print(f"Target object has {len(gaussian_positions)} Gaussian splats")
        print(f"Sample Gaussian positions: {gaussian_positions[:3]}")
        
        # Find point cloud points that are close to target Gaussians
        full_points = np.asarray(self.full_pcd.points)
        full_colors = np.asarray(self.full_pcd.colors) if len(self.full_pcd.colors) > 0 else None
        
        print(f"Sample point cloud positions: {full_points[:3]}")
        print(f"Point cloud range: X[{full_points[:,0].min():.3f}, {full_points[:,0].max():.3f}], Y[{full_points[:,1].min():.3f}, {full_points[:,1].max():.3f}], Z[{full_points[:,2].min():.3f}, {full_points[:,2].max():.3f}]")
        print(f"Gaussian range: X[{gaussian_positions[:,0].min():.3f}, {gaussian_positions[:,0].max():.3f}], Y[{gaussian_positions[:,1].min():.3f}, {gaussian_positions[:,1].max():.3f}], Z[{gaussian_positions[:,2].min():.3f}, {gaussian_positions[:,2].max():.3f}]")
        
        # Check if coordinate systems match
        gaussian_center = gaussian_positions.mean(axis=0)
        pcd_center = full_points.mean(axis=0)
        print(f"Gaussian center: {gaussian_center}")
        print(f"Point cloud center: {pcd_center}")
        
        # Use KD-tree to find nearest point cloud points to Gaussians
        from sklearn.neighbors import NearestNeighbors
        
        # Find point cloud points within a threshold distance of any Gaussian
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(full_points)
        distances, indices = nbrs.kneighbors(gaussian_positions)
        
        print(f"Distance statistics: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}, median={np.median(distances):.6f}")
        
        # Use more generous threshold - either adaptive or fixed
        if distances.min() < 1e-6:  # Very close matches exist
            threshold = np.percentile(distances, 90)  # Use 90th percentile
        else:
            # If no very close matches, use a more generous fixed threshold
            threshold = max(0.01, np.percentile(distances, 50))  # At least 1cm or median distance
        
        valid_indices = indices[distances.flatten() < threshold].flatten()
        valid_indices = np.unique(valid_indices)  # Remove duplicates
        
        print(f"Extracted {len(valid_indices)} point cloud points for target object (threshold: {threshold:.6f})")
        
        # If still no points found, try a much larger threshold
        if len(valid_indices) == 0:
            threshold = np.percentile(distances, 25)  # Use 25th percentile
            valid_indices = indices[distances.flatten() < threshold].flatten()
            valid_indices = np.unique(valid_indices)
            print(f"Retry with larger threshold: {len(valid_indices)} points (threshold: {threshold:.6f})")
        
        # If still no points, use spatial proximity approach
        if len(valid_indices) == 0:
            print("No points found with distance-based approach, trying spatial proximity...")
            # Find points within bounding box of Gaussians, expanded by some margin
            margin = 0.05  # 5cm margin
            min_bounds = gaussian_positions.min(axis=0) - margin
            max_bounds = gaussian_positions.max(axis=0) + margin
            
            mask = ((full_points >= min_bounds) & (full_points <= max_bounds)).all(axis=1)
            valid_indices = np.where(mask)[0]
            print(f"Found {len(valid_indices)} points using bounding box approach")
        
        # Create new point cloud with only target object points
        if len(valid_indices) > 0:
            target_points = full_points[valid_indices]
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            
            if full_colors is not None and len(valid_indices) <= len(full_colors):
                target_colors = full_colors[valid_indices]
                target_pcd.colors = o3d.utility.Vector3dVector(target_colors)
            
            # Calculate and store original centroid
            self.original_pcd_centroid = target_points.mean(axis=0)
            print(f"Original target object point cloud centroid: {self.original_pcd_centroid}")
            
            return target_pcd
        else:
            raise ValueError(f"No point cloud points found for target object '{self.optimizer.pipeline.model.__class__.__name__}' at index {self.object_idx}. "
                           f"The Gaussian splats and point cloud may be in different coordinate systems or the object segmentation failed.")
    
    def _extract_static_objects_pointcloud(self) -> o3d.geometry.PointCloud:
        """Extract point cloud points for all static objects (everything except target object)."""
        # Get all point cloud points
        full_points = np.asarray(self.full_pcd.points)
        full_colors = np.asarray(self.full_pcd.colors) if len(self.full_pcd.colors) > 0 else None
        
        # Get target object points to exclude them
        target_points = np.asarray(self.target_object_pcd.points)
        
        if len(target_points) == 0:
            # If no target points, return full point cloud as static
            return self.full_pcd
        
        # Find indices of points that are NOT part of the target object
        # Use spatial matching to find which full points correspond to target points
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
        distances, _ = nbrs.kneighbors(full_points)
        
        # Points with distance > small threshold are considered static
        static_mask = distances.flatten() > 1e-6
        static_indices = np.where(static_mask)[0]
        
        print(f"Static objects have {len(static_indices)} point cloud points")
        
        # Create static objects point cloud
        static_points = full_points[static_indices]
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(static_points)
        
        if full_colors is not None and len(static_indices) <= len(full_colors):
            static_colors = full_colors[static_indices]
            static_pcd.colors = o3d.utility.Vector3dVector(static_colors)
        
        return static_pcd
    
    def update_object_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Update the pose of the specified object in the optimizer (for Gaussians)."""
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
    
    def transform_target_pointcloud(self, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Transform the target object's point cloud to the specified position and orientation."""
        # Convert orientation to rotation matrix
        rot_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        rotation_matrix = R.from_quat(rot_xyzw).as_matrix()
        
        # Get target object points
        points = np.asarray(self.target_object_pcd.points)
        
        # Center the point cloud at origin (relative to original centroid)
        centered_points = points - self.original_pcd_centroid
        
        # Apply rotation
        rotated_points = centered_points @ rotation_matrix.T
        
        # Translate to target position
        final_points = rotated_points + position
        
        # Calculate new centroid
        new_centroid = final_points.mean(axis=0)
        
        return new_centroid
    
    def generate_transformed_target_pointcloud(self, position: np.ndarray, orientation: np.ndarray) -> o3d.geometry.PointCloud:
        """Generate the actual transformed point cloud for the target object."""
        # Convert orientation to rotation matrix
        rot_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        rotation_matrix = R.from_quat(rot_xyzw).as_matrix()
        
        # Get target object points
        points = np.asarray(self.target_object_pcd.points)
        colors = np.asarray(self.target_object_pcd.colors) if len(self.target_object_pcd.colors) > 0 else None
        
        # Center the point cloud at origin (relative to original centroid)
        centered_points = points - self.original_pcd_centroid
        
        # Apply rotation
        rotated_points = centered_points @ rotation_matrix.T
        
        # Translate to target position
        final_points = rotated_points + position
        
        # Create transformed point cloud
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(final_points)
        
        if colors is not None:
            transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return transformed_pcd
    
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
    
    def animate(self, camera, duration: float = 10.0, fps: int = 30) -> List[np.ndarray]:
        """Generate animation frames for video output."""
        total_frames = int(duration * fps)
        frames = []
        
        print(f"Generating {total_frames} frames for video...")
        
        for frame_idx in range(total_frames):
            t = frame_idx / (total_frames - 1) if total_frames > 1 else 0
            
            # Calculate which waypoint segment we're in
            num_segments = len(self.waypoints) - 1
            if num_segments == 0:
                current_pos = self.waypoints[0]
                current_orient = self.orientations[0]
            else:
                segment_duration = 1.0 / num_segments
                segment_idx = int(t / segment_duration)
                segment_t = (t - segment_idx * segment_duration) / segment_duration
                
                # Ensure we don't go out of bounds
                segment_idx = min(segment_idx, num_segments - 1)
                
                # Linear interpolation for position
                start_pos = self.waypoints[segment_idx]
                end_pos = self.waypoints[segment_idx + 1]
                current_pos = start_pos + segment_t * (end_pos - start_pos)
                
                # SLERP for orientation
                start_orient = self.orientations[segment_idx]
                end_orient = self.orientations[segment_idx + 1]
                
                # Convert to scipy Rotation objects for SLERP
                rot_start = R.from_quat([start_orient[1], start_orient[2], start_orient[3], start_orient[0]])
                rot_end = R.from_quat([end_orient[1], end_orient[2], end_orient[3], end_orient[0]])
                
                # Create slerp interpolator
                slerp = Slerp([0, 1], R.from_quat([rot_start.as_quat(), rot_end.as_quat()]))
                interpolated = slerp(segment_t)
                
                # Convert back to wxyz format
                quat_xyzw = interpolated.as_quat()
                current_orient = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            # Update object pose
            self.update_object_pose(current_pos, current_orient)
            
            # Render frame
            frame = self.render_frame(camera)
            frames.append(frame)
            
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")
        
        # Reset to original pose
        self.optimizer.optimizer.part_deltas = self.original_part_deltas.clone()
        
        return frames
    
    def save_scene_at_waypoint(self, waypoint_idx: int, position: np.ndarray, orientation: np.ndarray, output_base_dir: str = "outputs", timestamp: str = None):
        """Save complete scene point cloud data at a specific waypoint."""
        from pathlib import Path
        import os
        from datetime import datetime
        
        # Use provided timestamp or generate new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create output directory structure
        scene_dir = Path(output_base_dir) / f"scene_animation_{timestamp}" / f"subgoal_{waypoint_idx+1}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving scene data to: {scene_dir}")
        
        # Generate transformed target object point cloud
        transformed_target_pcd = self.generate_transformed_target_pointcloud(position, orientation)
        
        # Save individual component point clouds
        target_path = scene_dir / "moving_object.ply"
        static_path = scene_dir / "static_objects.ply"
        combined_path = scene_dir / "complete_scene.ply"
        
        # Save moving object (transformed)
        o3d.io.write_point_cloud(str(target_path), transformed_target_pcd)
        print(f"  Saved moving object ({len(transformed_target_pcd.points)} points): {target_path.name}")
        
        # Save static objects (unchanged)
        o3d.io.write_point_cloud(str(static_path), self.static_objects_pcd)
        print(f"  Saved static objects ({len(self.static_objects_pcd.points)} points): {static_path.name}")
        
        # Create and save combined scene
        combined_pcd = o3d.geometry.PointCloud()
        
        # Combine points
        transformed_points = np.asarray(transformed_target_pcd.points)
        static_points = np.asarray(self.static_objects_pcd.points)
        all_points = np.vstack([transformed_points, static_points])
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        
        # Combine colors if available
        if len(transformed_target_pcd.colors) > 0 and len(self.static_objects_pcd.colors) > 0:
            transformed_colors = np.asarray(transformed_target_pcd.colors)
            static_colors = np.asarray(self.static_objects_pcd.colors)
            all_colors = np.vstack([transformed_colors, static_colors])
            combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        o3d.io.write_point_cloud(str(combined_path), combined_pcd)
        print(f"  Saved complete scene ({len(combined_pcd.points)} points): {combined_path.name}")
        
        # Save metadata
        metadata = {
            "subgoal_index": waypoint_idx + 1,
            "target_waypoint": position.tolist(),
            "orientation_wxyz": orientation.tolist(),
            "moving_object_points": len(transformed_target_pcd.points),
            "static_objects_points": len(self.static_objects_pcd.points),
            "total_scene_points": len(combined_pcd.points),
            "moving_object_centroid": transformed_target_pcd.get_center().tolist(),
            "static_objects_centroid": self.static_objects_pcd.get_center().tolist()
        }
        
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path.name}")
        
        return scene_dir
    
    def generate_and_save_scene_animation(self, target_object_name: str, output_base_dir: str = "outputs"):
        """Generate and save complete scene point cloud data at each waypoint."""
        from datetime import datetime
        
        print(f"\n=== {target_object_name.upper()} SCENE ANIMATION GENERATION ===")
        print(f"Original Gaussian position:    [{self.original_position[0]:8.6f}, {self.original_position[1]:8.6f}, {self.original_position[2]:8.6f}]")
        print(f"Original PointCloud centroid:  [{self.original_pcd_centroid[0]:8.6f}, {self.original_pcd_centroid[1]:8.6f}, {self.original_pcd_centroid[2]:8.6f}]")
        print(f"Target object has {len(self.target_object_pcd.points)} point cloud points")
        print(f"Static objects have {len(self.static_objects_pcd.points)} point cloud points")
        print(f"Generating scene data for {len(self.waypoints)} waypoints...")
        print()
        
        # Generate single timestamp for all waypoints
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        saved_dirs = []
        
        for i, waypoint in enumerate(self.waypoints):
            # Get corresponding orientation
            orientation = self.orientations[i]
            
            # Update Gaussian pose (this also applies transformations)
            self.update_object_pose(waypoint, orientation)
            
            # Apply the transformations to the model to get updated Gaussian positions
            self.optimizer.optimizer.apply_to_model(
                self.optimizer.optimizer.part_deltas,
                self.optimizer.optimizer.group_labels
            )
            
            # Get current Gaussian position after transformation
            current_gaussian_pose = self.optimizer.get_parts2world()[self.object_idx]
            current_gaussian_pos = current_gaussian_pose.translation()
            
            # Transform target object point cloud
            transformed_pcd_centroid = self.transform_target_pointcloud(waypoint, orientation)
            
            # Calculate deltas
            gaussian_delta = current_gaussian_pos - self.original_position
            pcd_delta = transformed_pcd_centroid - self.original_pcd_centroid
            
            print(f"Subgoal {i+1}/{len(self.waypoints)}:")
            print(f"  Target waypoint:        [{waypoint[0]:8.6f}, {waypoint[1]:8.6f}, {waypoint[2]:8.6f}]")
            print(f"  Gaussian position:      [{current_gaussian_pos[0]:8.6f}, {current_gaussian_pos[1]:8.6f}, {current_gaussian_pos[2]:8.6f}]")
            print(f"  PointCloud centroid:    [{transformed_pcd_centroid[0]:8.6f}, {transformed_pcd_centroid[1]:8.6f}, {transformed_pcd_centroid[2]:8.6f}]")
            print(f"  Gaussian delta:         [{gaussian_delta[0]:8.6f}, {gaussian_delta[1]:8.6f}, {gaussian_delta[2]:8.6f}]")
            print(f"  PointCloud delta:       [{pcd_delta[0]:8.6f}, {pcd_delta[1]:8.6f}, {pcd_delta[2]:8.6f}]")
            print(f"  Gaussian distance:      {np.linalg.norm(gaussian_delta):8.6f}")
            print(f"  PointCloud distance:    {np.linalg.norm(pcd_delta):8.6f}")
            
            # Save scene data at this waypoint
            scene_dir = self.save_scene_at_waypoint(i, waypoint, orientation, output_base_dir, timestamp)
            saved_dirs.append(scene_dir)
            print()
        
        # Reset to original pose
        self.optimizer.optimizer.part_deltas = self.original_part_deltas.clone()
        
        # Summary
        total_distance = 0
        for i in range(1, len(self.waypoints)):
            distance = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1])
            total_distance += distance
        
        final_gaussian_delta = self.waypoints[-1] - self.original_position
        final_pcd_delta = self.waypoints[-1] - self.original_pcd_centroid
        
        print("=== SCENE GENERATION SUMMARY ===")
        print(f"Total waypoints:               {len(self.waypoints)}")
        print(f"Scene directories created:     {len(saved_dirs)}")
        print(f"Total path distance:           {total_distance:.6f}")
        print(f"Final Gaussian displacement:   {np.linalg.norm(final_gaussian_delta):.6f}")
        print(f"Final PointCloud displacement: {np.linalg.norm(final_pcd_delta):.6f}")
        print(f"Final waypoint:                [{self.waypoints[-1][0]:8.6f}, {self.waypoints[-1][1]:8.6f}, {self.waypoints[-1][2]:8.6f}]")
        print(f"\nScene data saved to:")
        for scene_dir in saved_dirs:
            print(f"  {scene_dir}")
        
        return saved_dirs


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


def load_waypoints_from_json(waypoints_file: str) -> List[Tuple[float, float, float]]:
    """Load waypoints from JSON file."""
    with open(waypoints_file, 'r') as f:
        data = json.load(f)
    
    # Extract waypoints (positions only) from all subgoal_pose
    waypoints = []
    for subgoal in data['subgoals']:
        subgoal_pose = subgoal['subgoal_pose']
        # Extract only the position (first 3 elements: x, y, z)
        waypoints.append((subgoal_pose[0], subgoal_pose[1], subgoal_pose[2]))
    
    return waypoints


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
    np.random.seed(42)
    random.seed(42)
    
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


def save_pose_data(animator: GaussianPointCloudAnimator, camera, position: np.ndarray, 
                   orientation: np.ndarray, pose_idx: int, output_base_dir: str = "outputs",
                   timestamp: str = None) -> Path:
    """Save both Gaussian render and point cloud data for a specific pose.
    
    Args:
        animator: The GaussianPointCloudAnimator instance
        camera: Camera for rendering Gaussian images
        position: Object position
        orientation: Object orientation quaternion [w, x, y, z]
        pose_idx: Pose index for naming
        output_base_dir: Base output directory
        timestamp: Timestamp for directory naming
        
    Returns:
        Path to the saved pose directory
    """
    from datetime import datetime
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output directory for this pose
    pose_dir = Path(output_base_dir) / f"random_poses_{timestamp}" / f"pose_{pose_idx:03d}"
    pose_dir.mkdir(parents=True, exist_ok=True)
    
    # Update object pose
    animator.update_object_pose(position, orientation)
    
    # Render Gaussian image
    gaussian_image = animator.render_frame(camera)
    
    # Save Gaussian render as PNG
    gaussian_path = pose_dir / "gaussian_render.png"
    from PIL import Image
    gaussian_pil = Image.fromarray(gaussian_image)
    gaussian_pil.save(gaussian_path)
    
    # Generate and save point cloud data
    transformed_target_pcd = animator.generate_transformed_target_pointcloud(position, orientation)
    
    # Save individual point clouds
    target_pcd_path = pose_dir / "moving_object.ply"
    static_pcd_path = pose_dir / "static_objects.ply"
    combined_pcd_path = pose_dir / "complete_scene.ply"
    
    # Save moving object point cloud
    o3d.io.write_point_cloud(str(target_pcd_path), transformed_target_pcd)
    
    # Save static objects point cloud
    o3d.io.write_point_cloud(str(static_pcd_path), animator.static_objects_pcd)
    
    # Create and save combined scene
    combined_pcd = o3d.geometry.PointCloud()
    
    # Combine points
    transformed_points = np.asarray(transformed_target_pcd.points)
    static_points = np.asarray(animator.static_objects_pcd.points)
    all_points = np.vstack([transformed_points, static_points])
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Combine colors if available
    if len(transformed_target_pcd.colors) > 0 and len(animator.static_objects_pcd.colors) > 0:
        transformed_colors = np.asarray(transformed_target_pcd.colors)
        static_colors = np.asarray(animator.static_objects_pcd.colors)
        all_colors = np.vstack([transformed_colors, static_colors])
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    o3d.io.write_point_cloud(str(combined_pcd_path), combined_pcd)
    
    # Save metadata
    metadata = {
        "pose_index": pose_idx,
        "position": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
        "moving_object_points": len(transformed_target_pcd.points),
        "static_objects_points": len(animator.static_objects_pcd.points),
        "total_scene_points": len(combined_pcd.points),
        "moving_object_centroid": transformed_target_pcd.get_center().tolist(),
        "files": {
            "gaussian_render": "gaussian_render.png",
            "moving_object": "moving_object.ply",
            "static_objects": "static_objects.ply",
            "complete_scene": "complete_scene.ply"
        }
    }
    
    metadata_path = pose_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return pose_dir


def main(
    config_path: Path = Path("outputs/box/pogs/2025-07-29_151651/config.yml"),
    pointcloud_path: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/box/prime_seg_gaussians.ply",
    semantic_query: str = "box cutter",
    waypoints_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/all_subgoals.json",
    generate_video: bool = True,
    animation_duration: float = 5.0,
    output_video_path: str = "gaussian_pointcloud_animation.mp4",
    fps: int = 30,
    show_all_objects: bool = True,
    calibration_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/src/pogs/calibration_outputs/world_to_d405.tf",
    enable_random_poses: bool = True,
    num_random_poses: int = 60,
    max_translation: float = 0.05,
    max_rotation_deg: float = 180.0,
    min_z: float = 0.005,
):
    """Analyze both Gaussian splats and point cloud movement for a target object, with optional video generation.
    
    Args:
        config_path: Path to POGS config file
        pointcloud_path: Path to the PLY point cloud file
        semantic_query: Semantic description of target object (e.g., "box cutter")
        waypoints_file: Path to JSON file containing waypoints
        generate_video: Whether to generate MP4 animation video
        animation_duration: Total animation duration in seconds (for video)
        output_video_path: Path for output video file
        fps: Frames per second for video output
        show_all_objects: Whether to show all objects in video or only target object
        calibration_file: Path to camera calibration file
        enable_random_poses: Whether to generate random poses after waypoint processing
        num_random_poses: Number of random poses to generate
        max_translation: Maximum translation per axis in meters
        max_rotation_deg: Maximum rotation per axis in degrees
        min_z: Minimum z-coordinate (above table surface)
    """
    # Initialize
    wp.init()
    print("Loading model...")
    
    # Load waypoints from JSON file
    waypoints = load_waypoints_from_json(waypoints_file)
    print(f"Loaded {len(waypoints)} waypoints from {waypoints_file}")
    
    # Setup optimizer and find target object
    optimizer = setup_optimizer(config_path)
    target_object_idx = find_target_object(optimizer, semantic_query)
    
    # Create combined animator
    animator = GaussianPointCloudAnimator(
        optimizer=optimizer,
        pointcloud_path=pointcloud_path,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=None  # Use original object's quaternion
    )
    
    print("\nGenerating scene animation data...")
    # Generate and save complete scene data at each waypoint
    saved_directories = animator.generate_and_save_scene_animation(semantic_query)
    
    # Generate random poses if requested
    random_pose_dirs = []
    if enable_random_poses:
        print(f"\n=== GENERATING RANDOM POSES ===")
        
        # Use the final waypoint as the base pose for random generation
        final_waypoint = waypoints[-1]
        base_position = np.array(final_waypoint)
        
        # Use the original object orientation as base orientation
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
        
        # Create camera for rendering
        camera = create_camera_from_calibration(calibration_file, optimizer)
        
        # Generate timestamp for random poses
        from datetime import datetime
        random_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        print(f"\nSaving random pose data...")
        for i, (pos, orient) in enumerate(zip(random_positions, random_orientations)):
            pose_dir = save_pose_data(
                animator=animator,
                camera=camera,
                position=pos,
                orientation=orient,
                pose_idx=i,
                output_base_dir="outputs",
                timestamp=random_timestamp
            )
            random_pose_dirs.append(pose_dir)
            
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{num_random_poses} poses")
        
        # Reset to original pose
        animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()
        
        print(f"\n=== RANDOM POSE GENERATION COMPLETE ===")
        print(f"Generated {len(random_pose_dirs)} random poses")
        print(f"Random pose data saved to: outputs/random_poses_{random_timestamp}/")
        
        # Summary of generated poses
        distances = [np.linalg.norm(pos - base_position) for pos in random_positions]
        print(f"Translation distances: min={min(distances):.4f}m, max={max(distances):.4f}m, avg={np.mean(distances):.4f}m")
        
        z_values = [pos[2] for pos in random_positions]
        print(f"Z-coordinates: min={min(z_values):.4f}m, max={max(z_values):.4f}m")
    
    # Generate video if requested
    if generate_video:
        print(f"\nGenerating MP4 animation video...")
        
        # Create camera from calibration
        camera = create_camera_from_calibration(calibration_file, optimizer)
        
        # Optionally hide other objects for cleaner visualization
        original_opacities = None
        if not show_all_objects:
            original_opacities = optimizer.pipeline.model.gauss_params["opacities"].clone()
            group_masks_global = optimizer.optimizer.group_masks  # Use the correct attribute
            for idx, mask in enumerate(group_masks_global):
                if idx != target_object_idx:
                    optimizer.pipeline.model.gauss_params["opacities"][mask] = -10.0
        
        # Generate animation frames
        frames = animator.animate(camera, duration=animation_duration, fps=fps)
        
        # Save video
        print(f"Saving video to {output_video_path}...")
        out_clip = mpy.ImageSequenceClip(frames, fps=fps)
        out_clip.write_videofile(output_video_path, codec='libx264')
        
        # Restore original opacities
        if original_opacities is not None:
            optimizer.pipeline.model.gauss_params["opacities"] = original_opacities
        
        print(f"Video generation complete! Saved to {output_video_path}")
    
    print(f"\nAnimation complete!")
    print(f"Point cloud scene data saved to {len(saved_directories)} directories")
    if enable_random_poses:
        print(f"Random pose data saved to {len(random_pose_dirs)} directories")
    if generate_video:
        print(f"MP4 video saved to: {output_video_path}")


if __name__ == "__main__":
    tyro.cli(main)