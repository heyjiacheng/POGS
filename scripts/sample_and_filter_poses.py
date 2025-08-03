#!/usr/bin/env python3
"""
Combined Gaussian animation, collision filtering, and pose ranking script.

This script provides a complete pipeline for:
1. Generating Gaussian splat and point cloud animations
2. Sampling and filtering random poses for target objects
3. Ranking poses by rotation similarity to final waypoint
4. Interactive pose selection and new subgoals generation
5. Creating new animations with selected poses

The script handles the relationship between end effector poses (stored in subgoals JSON)
and target object poses (used for animation and visualization).
"""

import torch
import numpy as np
import tyro
import time
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import random
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial import cKDTree

import warp as wp
import moviepy as mpy
import trimesh
import open3d as o3d

from pogs.tracking.optim import Optimizer
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
from nerfstudio.cameras.cameras import Cameras


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


# =============================================================================
# COLLISION DETECTION
# =============================================================================

class CollisionDetector:
    """SDF-based collision detection between point clouds."""
    
    def __init__(self, collision_threshold: float = 0.001):
        """
        Initialize collision detector.
        
        Args:
            collision_threshold: Minimum distance threshold for collision detection (meters)
        """
        self.collision_threshold = collision_threshold
    
    def compute_sdf(self, query_points: np.ndarray, reference_points: np.ndarray, 
                   k_neighbors: int = 5) -> np.ndarray:
        """
        Compute signed distance field values for query points relative to reference points.
        
        Args:
            query_points: Points to query SDF values for (N, 3)
            reference_points: Reference point cloud (M, 3)
            k_neighbors: Number of neighbors to use for SDF computation
            
        Returns:
            SDF values for each query point (N,)
        """
        if len(reference_points) == 0:
            return np.full(len(query_points), np.inf)
        
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(reference_points)
        
        # Find k nearest neighbors for each query point
        distances, _ = tree.query(query_points, k=min(k_neighbors, len(reference_points)))
        
        # If only one neighbor, distances is 1D, make it 2D
        if k_neighbors == 1 or len(reference_points) == 1:
            distances = distances.reshape(-1, 1)
        
        # Use minimum distance to reference points as unsigned distance
        min_distances = distances[:, 0]
        
        return min_distances
    
    def detect_collision_sdf(self, moving_pcd: o3d.geometry.PointCloud, 
                            static_pcd: o3d.geometry.PointCloud) -> Tuple[bool, float, Dict]:
        """
        Detect collision between moving and static point clouds using SDF.
        
        Args:
            moving_pcd: Moving object point cloud
            static_pcd: Static objects point cloud
            
        Returns:
            Tuple of (collision_detected, min_distance, collision_info)
        """
        moving_points = np.asarray(moving_pcd.points)
        static_points = np.asarray(static_pcd.points)
        
        if len(moving_points) == 0 or len(static_points) == 0:
            return False, np.inf, {"reason": "empty_pointcloud"}
        
        # Compute SDF values for moving object points relative to static objects
        sdf_values = self.compute_sdf(moving_points, static_points)
        
        # Find minimum distance
        min_distance = np.min(sdf_values)
        
        # Count collision points
        collision_points = np.sum(sdf_values < self.collision_threshold)
        collision_ratio = collision_points / len(moving_points)
        
        # Detect collision
        collision_detected = min_distance < self.collision_threshold
        
        collision_info = {
            "min_distance": float(min_distance),
            "collision_points": int(collision_points),
            "total_moving_points": len(moving_points),
            "collision_ratio": float(collision_ratio),
            "collision_threshold": self.collision_threshold,
            "method": "sdf"
        }
        
        return collision_detected, min_distance, collision_info


# =============================================================================
# GAUSSIAN POINT CLOUD ANIMATOR
# =============================================================================

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
    
    def _remove_outliers(self, pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
        """Remove outliers from point cloud using statistical outlier removal.
        
        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors to consider for outlier detection
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            Point cloud with outliers removed
        """
        if len(pcd.points) == 0:
            return pcd
        
        original_count = len(pcd.points)
        
        # Statistical outlier removal
        pcd_filtered, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
        removed_count = original_count - len(pcd_filtered.points)
        if removed_count > 0:
            print(f"  Removed {removed_count} outliers ({removed_count/original_count:.1%}) from point cloud")
        
        return pcd_filtered
    
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
            
            # Remove outliers from target object point cloud
            print(f"Removing outliers from target object point cloud...")
            target_pcd = self._remove_outliers(target_pcd)
            
            # Recalculate points after outlier removal
            target_points = np.asarray(target_pcd.points)
            
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
        
        # Remove outliers from static objects point cloud
        print(f"Removing outliers from static objects point cloud...")
        static_pcd = self._remove_outliers(static_pcd)
        
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


# =============================================================================
# WAYPOINT AND POSE MANAGEMENT
# =============================================================================

def load_waypoints_from_json(waypoints_file: str) -> List[Tuple[float, float, float]]:
    """Load waypoints from JSON file.
    
    Note: These are end effector waypoints, not target object waypoints.
    For target object animation, we need to use the target object's own initial pose.
    """
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


def calculate_rotation_difference(quat1: np.ndarray, quat2: np.ndarray) -> float:
    """Calculate the rotation difference between two quaternions in degrees.
    
    Args:
        quat1: First quaternion [w, x, y, z]
        quat2: Second quaternion [w, x, y, z]
        
    Returns:
        Rotation difference in degrees
    """
    # Convert to scipy Rotation objects
    rot1_xyzw = np.array([quat1[1], quat1[2], quat1[3], quat1[0]])
    rot2_xyzw = np.array([quat2[1], quat2[2], quat2[3], quat2[0]])
    
    rot1 = R.from_quat(rot1_xyzw)
    rot2 = R.from_quat(rot2_xyzw)
    
    # Calculate relative rotation
    relative_rotation = rot2 * rot1.inv()
    
    # Get rotation angle in radians and convert to degrees
    angle_rad = relative_rotation.magnitude()
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


# =============================================================================
# INTERACTIVE POSE SELECTION
# =============================================================================

def interactive_pose_selection(pose_rankings: List[Tuple[Any, float]], max_display: int = 10) -> Optional[int]:
    """Interactive pose selection from ranked poses.
    
    Args:
        pose_rankings: List of (pose_data, rotation_difference) tuples
        max_display: Maximum number of poses to display
        
    Returns:
        Selected pose index (1-based) or None if 0 is selected
    """
    print(f"\n=== INTERACTIVE POSE SELECTION ===")
    print(f"Available poses ranked by rotation similarity to final waypoint:")
    print(f"(Smaller rotation differences = better matches)\n")
    
    # Display top poses
    display_count = min(max_display, len(pose_rankings))
    for i in range(display_count):
        pose_data, rot_diff = pose_rankings[i]
        print(f"  {i+1:2d}. Pose {pose_data['index']:03d}: {rot_diff:.2f}° rotation difference")
    
    if len(pose_rankings) > max_display:
        print(f"  ... and {len(pose_rankings) - max_display} more poses available")
    
    print(f"\nOptions:")
    print(f"  0: Keep original waypoints (no replacement)")
    print(f"  1-{len(pose_rankings)}: Select pose to replace final waypoint")
    
    # Get user input
    while True:
        try:
            choice = input(f"\nEnter your choice (0-{len(pose_rankings)}): ").strip()
            choice_int = int(choice)
            
            if 0 <= choice_int <= len(pose_rankings):
                if choice_int == 0:
                    print("Keeping original waypoints.")
                    return None
                else:
                    selected_pose = pose_rankings[choice_int - 1]
                    print(f"Selected pose {choice_int}: {selected_pose[1]:.2f}° rotation difference")
                    return choice_int
            else:
                print(f"Please enter a number between 0 and {len(pose_rankings)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None


# =============================================================================
# END EFFECTOR TO TARGET OBJECT CONVERSION
# =============================================================================

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


def convert_ee_waypoints_to_target_waypoints(ee_waypoints_file: str, 
                                            original_target_pose: np.ndarray, 
                                            original_target_orient: np.ndarray) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float, float]]]:
    """Convert end effector waypoints to target object waypoints for animation.
    
    The key insight is that we need to maintain the relative relationship between
    end effector and target object throughout the motion.
    
    Args:
        ee_waypoints_file: Path to end effector waypoints JSON file
        original_target_pose: Original target object position
        original_target_orient: Original target object orientation
        
    Returns:
        Tuple of (target_waypoints, target_orientations)
    """
    with open(ee_waypoints_file, 'r') as f:
        data = json.load(f)
    
    target_waypoints = []
    target_orientations = []
    
    # Get the first end effector pose to calculate relative transform
    first_subgoal = data['subgoals'][0]['subgoal_pose']
    first_ee_pos = np.array(first_subgoal[:3])
    first_ee_orient = np.array([first_subgoal[6], first_subgoal[3], first_subgoal[4], first_subgoal[5]])  # [w, x, y, z]
    
    # Calculate the relative transform from first end effector to target object
    relative_pos, relative_orient = get_ee_target_relative_transform(
        first_ee_pos, first_ee_orient,
        original_target_pose, original_target_orient
    )
    
    print(f"\nCalculated relative transform from end effector to target object:")
    print(f"Relative position: [{relative_pos[0]:.6f}, {relative_pos[1]:.6f}, {relative_pos[2]:.6f}]")
    print(f"Relative orientation: [{relative_orient[0]:.6f}, {relative_orient[1]:.6f}, {relative_orient[2]:.6f}, {relative_orient[3]:.6f}]")
    
    # Apply the same relative transform to all end effector waypoints
    for subgoal in data['subgoals']:
        subgoal_pose = subgoal['subgoal_pose']
        ee_pos = np.array(subgoal_pose[:3])  # [x, y, z]
        ee_orient = np.array([subgoal_pose[6], subgoal_pose[3], subgoal_pose[4], subgoal_pose[5]])  # [w, x, y, z]
        
        # Apply relative transform to get target object pose
        target_pos, target_orient = apply_relative_transform(
            ee_pos, ee_orient,
            relative_pos, relative_orient
        )
        
        target_waypoints.append((target_pos[0], target_pos[1], target_pos[2]))
        target_orientations.append((target_orient[0], target_orient[1], target_orient[2], target_orient[3]))
    
    print(f"\nTarget waypoint conversion:")
    print(f"Original target pose: [{original_target_pose[0]:.6f}, {original_target_pose[1]:.6f}, {original_target_pose[2]:.6f}]")
    print(f"First target pose:    [{target_waypoints[0][0]:.6f}, {target_waypoints[0][1]:.6f}, {target_waypoints[0][2]:.6f}]")
    print(f"Final target pose:    [{target_waypoints[-1][0]:.6f}, {target_waypoints[-1][1]:.6f}, {target_waypoints[-1][2]:.6f}]")
    
    return target_waypoints, target_orientations


def create_new_subgoals_json(original_waypoints_file: str, selected_pose_data: Optional[Dict], 
                             original_target_pose: np.ndarray, original_target_orient: np.ndarray,
                             output_file: str = "outputs/new_subgoals.json") -> str:
    """Create new subgoals JSON file with selected pose replacing final waypoint.
    
    Args:
        original_waypoints_file: Path to original waypoints JSON file
        selected_pose_data: Selected pose data dict or None to keep original
        original_target_pose: Original target object position
        original_target_orient: Original target object orientation
        output_file: Output path for new subgoals JSON
        
    Returns:
        Path to created JSON file
    """
    # Load original waypoints
    with open(original_waypoints_file, 'r') as f:
        original_data = json.load(f)
    
    # Create new data structure
    new_data = original_data.copy()
    
    if selected_pose_data is not None:
        # Get the first end effector pose to calculate relative transform
        first_subgoal = original_data['subgoals'][0]['subgoal_pose']
        first_ee_pos = np.array(first_subgoal[:3])
        first_ee_orient = np.array([first_subgoal[6], first_subgoal[3], first_subgoal[4], first_subgoal[5]])  # [w, x, y, z]
        
        # Calculate the relative transform from first end effector to original target object
        relative_pos, relative_orient = get_ee_target_relative_transform(
            first_ee_pos, first_ee_orient,
            original_target_pose, original_target_orient
        )
        
        # Extract new target object pose
        new_target_pose = selected_pose_data['position']
        new_target_orient = selected_pose_data['orientation']  # [w, x, y, z]
        
        # Calculate new end effector pose that maintains the same relative transform
        # We need to find ee_pose such that: apply_relative_transform(ee_pose, relative) = new_target_pose
        
        # Convert to rotation objects
        new_target_rot = R.from_quat([new_target_orient[1], new_target_orient[2], 
                                     new_target_orient[3], new_target_orient[0]])
        relative_rot = R.from_quat([relative_orient[1], relative_orient[2], 
                                   relative_orient[3], relative_orient[0]])
        
        # Calculate required end effector rotation
        new_ee_rot = new_target_rot * relative_rot.inv()
        new_ee_quat = new_ee_rot.as_quat()  # [x, y, z, w]
        new_ee_orientation = np.array([new_ee_quat[3], new_ee_quat[0], new_ee_quat[1], new_ee_quat[2]])  # [w, x, y, z]
        
        # Calculate required end effector position
        new_ee_position = new_target_pose - new_ee_rot.apply(relative_pos)
        
        # Convert to pose format [x, y, z, qx, qy, qz, qw]
        new_ee_pose = [
            new_ee_position[0], new_ee_position[1], new_ee_position[2],  # position
            new_ee_orientation[1], new_ee_orientation[2], new_ee_orientation[3], new_ee_orientation[0]  # quaternion xyzw
        ]
        
        # Update final subgoal with new end effector pose
        final_subgoal_idx = len(new_data['subgoals']) - 1
        new_data['subgoals'][final_subgoal_idx]['subgoal_pose'] = new_ee_pose
        
        # Add metadata about the replacement
        new_data['metadata'] = {
            "modified": True,
            "original_file": original_waypoints_file,
            "replaced_final_waypoint": True,
            "selected_pose_index": selected_pose_data['index'],
            "rotation_difference_deg": selected_pose_data.get('rotation_difference_deg', 0.0),
            "modification_timestamp": datetime.now().isoformat(),
            "transformation_note": "End effector pose calculated to maintain relative transform to target object"
        }
        
        print(f"\n=== NEW SUBGOALS CREATED ===")
        print(f"Original target object pose: [{original_target_pose[0]:.6f}, {original_target_pose[1]:.6f}, {original_target_pose[2]:.6f}]")
        print(f"New target object pose:      [{new_target_pose[0]:.6f}, {new_target_pose[1]:.6f}, {new_target_pose[2]:.6f}]")
        
        # Get original final end effector pose for comparison
        original_final_subgoal = original_data['subgoals'][final_subgoal_idx]['subgoal_pose']
        original_final_ee_pos = np.array(original_final_subgoal[:3])
        print(f"Original end effector pose:  [{original_final_ee_pos[0]:.6f}, {original_final_ee_pos[1]:.6f}, {original_final_ee_pos[2]:.6f}]")
        print(f"New end effector pose:       [{new_ee_position[0]:.6f}, {new_ee_position[1]:.6f}, {new_ee_position[2]:.6f}]")
        
        # Verify the transformation is correct
        verify_target_pos, verify_target_orient = apply_relative_transform(
            new_ee_position, new_ee_orientation,
            relative_pos, relative_orient
        )
        print(f"Verification - computed target pose: [{verify_target_pos[0]:.6f}, {verify_target_pos[1]:.6f}, {verify_target_pos[2]:.6f}]")
        
        print(f"Selected pose index:         {selected_pose_data['index']}")
        if 'rotation_difference_deg' in selected_pose_data:
            print(f"Rotation difference:         {selected_pose_data['rotation_difference_deg']:.2f}°")
    else:
        # Keep original waypoints
        new_data['metadata'] = {
            "modified": False,
            "original_file": original_waypoints_file,
            "modification_timestamp": datetime.now().isoformat()
        }
        print(f"\n=== KEEPING ORIGINAL SUBGOALS ===")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save new subgoals
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"New subgoals saved to: {output_file}")
    return output_file


def generate_new_animation(existing_optimizer: Optimizer, existing_camera, pointcloud_path: str, 
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
    
    # Ensure output directory exists
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save video
    print(f"Saving new animation to {output_video_path}...")
    out_clip = mpy.ImageSequenceClip(frames, fps=fps)
    out_clip.write_videofile(output_video_path, codec='libx264')
    
    # Restore original opacities
    if original_opacities is not None:
        existing_optimizer.pipeline.model.gauss_params["opacities"] = original_opacities
    
    print(f"New animation generation complete! Saved to {output_video_path}")
    return output_video_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(
    config_path: Path = Path("outputs/box/pogs/2025-07-29_151651/config.yml"),
    pointcloud_path: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/box/prime_seg_gaussians.ply",
    semantic_query: str = "box cutter",
    waypoints_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/all_subgoals.json",
    generate_video: bool = True,
    animation_duration: float = 5.0,
    output_video_path: str = "outputs/test_videos/gaussian_pointcloud_animation.mp4",
    fps: int = 30,
    show_all_objects: bool = True,
    calibration_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/src/pogs/calibration_outputs/world_to_d405.tf",
    enable_random_poses: bool = True,
    num_random_poses: int = 1000,
    max_translation: float = 0.05,
    max_rotation_deg: float = 180.0,
    min_z: float = 0.005,
    collision_threshold: float = 0.001,
    enable_pose_ranking: bool = True,
    max_ranked_poses: Optional[int] = 20,
):
    """Combined Gaussian animation, collision filtering, and pose ranking.
    
    This is the main pipeline that:
    1. Loads the scene and creates animations for waypoints
    2. Generates random poses around the final waypoint
    3. Filters poses for collisions using SDF-based detection
    4. Ranks valid poses by rotation similarity to final waypoint
    5. Provides interactive selection of poses
    6. Creates new subgoals and animations based on selected pose
    
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
        collision_threshold: Collision detection threshold in meters
        enable_pose_ranking: Whether to rank non-collision poses by rotation
        max_ranked_poses: Maximum number of ranked poses to save (None for all)
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
    
    # Create camera early for potential reuse
    camera = None
    if generate_video or enable_random_poses:
        print(f"\nCreating camera from calibration...")
        camera = create_camera_from_calibration(calibration_file, optimizer)
    
    # Generate random poses if requested
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
        
        # Camera should already be created above, but safety check
        if camera is None:
            camera = create_camera_from_calibration(calibration_file, optimizer)
        
        # Generate timestamp for random poses
        from datetime import datetime
        random_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        print(f"\nProcessing random poses for collision detection and ranking...")
        
        # Initialize collision detector
        collision_detector = CollisionDetector(collision_threshold)
        
        # Process poses directly without saving intermediate directories
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
                # Store valid pose data
                pose_data = {
                    'position': pos,
                    'orientation': orient,
                    'index': pose_idx,
                    'transformed_pcd': transformed_target_pcd
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
        
        # Pose ranking and saving final results        
        if enable_pose_ranking and valid_pose_data:
            print(f"\n=== POSE RANKING ===")
            
            # Get final waypoint orientation for ranking
            final_orientation = base_orientation
            
            # Calculate rotation differences and create rankings
            pose_rankings = []
            for pose_data in valid_pose_data:
                rotation_diff = calculate_rotation_difference(final_orientation, pose_data['orientation'])
                pose_rankings.append((pose_data, rotation_diff))
            
            # Sort by rotation difference (ascending - smaller differences first)
            pose_rankings.sort(key=lambda x: x[1])
            
            print(f"Pose ranking complete. Top 10 poses with smallest rotation differences:")
            for i, (pose_data, rot_diff) in enumerate(pose_rankings[:10]):
                print(f"  {i+1:2d}. Pose {pose_data['index']:03d}: {rot_diff:.2f}°")
            
            if len(pose_rankings) > 10:
                print(f"  ... and {len(pose_rankings) - 10} more poses")
            
            # Save only the top ranked poses
            num_to_save = min(max_ranked_poses or len(pose_rankings), len(pose_rankings))
            ranked_output = f"outputs/ranked_poses_{random_timestamp}"
            ranked_path = Path(ranked_output)
            ranked_path.mkdir(parents=True, exist_ok=True)
            
            print(f"\nSaving top {num_to_save} ranked poses to {ranked_output}...")
            
            ranking_info = []
            for pose_idx, (pose_data, rot_diff) in enumerate(pose_rankings[:num_to_save]):
                # Create new pose directory name with ranking
                new_pose_name = f"ranked_pose_{pose_idx+1:03d}"
                dest_dir = ranked_path / new_pose_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Update object pose and render
                animator.update_object_pose(pose_data['position'], pose_data['orientation'])
                gaussian_image = animator.render_frame(camera)
                
                # Save Gaussian render as PNG
                gaussian_path = dest_dir / "gaussian_render.png"
                from PIL import Image
                gaussian_pil = Image.fromarray(gaussian_image)
                gaussian_pil.save(gaussian_path)
                
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
                
                # Save metadata
                metadata = {
                    "pose_index": pose_idx,
                    "original_pose_index": pose_data['index'],
                    "position": pose_data['position'].tolist(),
                    "orientation_wxyz": pose_data['orientation'].tolist(),
                    "moving_object_points": len(pose_data['transformed_pcd'].points),
                    "static_objects_points": len(animator.static_objects_pcd.points),
                    "total_scene_points": len(combined_pcd.points),
                    "moving_object_centroid": pose_data['transformed_pcd'].get_center().tolist(),
                    "ranking_position": pose_idx + 1,
                    "rotation_difference_deg": rot_diff,
                    "ranked_by_rotation": True,
                    "files": {
                        "gaussian_render": "gaussian_render.png",
                        "moving_object": "moving_object.ply",
                        "static_objects": "static_objects.ply",
                        "complete_scene": "complete_scene.ply"
                    }
                }
                
                metadata_path = dest_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                ranking_info.append({
                    "rank": pose_idx + 1,
                    "original_index": pose_data['index'],
                    "new_name": new_pose_name,
                    "rotation_difference_deg": rot_diff
                })
                
                if (pose_idx + 1) % 5 == 0:
                    print(f"  Saved {pose_idx + 1}/{num_to_save} poses")
            
            # Reset to original pose
            animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()
            
            # Save ranking summary
            ranking_summary = {
                "total_generated_poses": num_random_poses,
                "valid_poses_after_filtering": len(valid_pose_data),
                "collision_poses_filtered": collision_count,
                "total_ranked_poses": num_to_save,
                "ranking_criteria": "rotation_difference_from_final_waypoint",
                "ranking_timestamp": datetime.now().isoformat(),
                "collision_threshold": collision_threshold,
                "pose_rankings": ranking_info
            }
            
            summary_file = ranked_path / "ranking_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(ranking_summary, f, indent=2)
            
            print(f"Ranking complete! Top {num_to_save} poses saved to: {ranked_output}")
            print(f"Ranking summary saved to: {summary_file}")
            
            # Summary of generated poses
            distances = [np.linalg.norm(pose_data['position'] - base_position) for pose_data, _ in pose_rankings[:num_to_save]]
            print(f"\nTop {num_to_save} poses statistics:")
            print(f"Translation distances: min={min(distances):.4f}m, max={max(distances):.4f}m, avg={np.mean(distances):.4f}m")
            
            z_values = [pose_data['position'][2] for pose_data, _ in pose_rankings[:num_to_save]]
            print(f"Z-coordinates: min={min(z_values):.4f}m, max={max(z_values):.4f}m")
            
            rotation_diffs = [rot_diff for _, rot_diff in pose_rankings[:num_to_save]]
            print(f"Rotation differences: min={min(rotation_diffs):.2f}°, max={max(rotation_diffs):.2f}°, avg={np.mean(rotation_diffs):.2f}°")
            
            # Interactive pose selection
            print(f"\n=== INTERACTIVE POSE SELECTION ===")
            selected_pose_idx = interactive_pose_selection(pose_rankings)
            
            if selected_pose_idx is not None:
                # User selected a pose to replace final waypoint
                selected_pose_data = pose_rankings[selected_pose_idx - 1][0]  # Get pose_data
                selected_pose_data['rotation_difference_deg'] = pose_rankings[selected_pose_idx - 1][1]  # Add rotation diff
                
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
                    # Fallback to identity quaternion if rotation format is unexpected
                    original_target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
                
                # Create new subgoals JSON with proper end effector pose transformation
                new_subgoals_file = create_new_subgoals_json(
                    original_waypoints_file=waypoints_file,
                    selected_pose_data=selected_pose_data,
                    original_target_pose=original_target_pose,
                    original_target_orient=original_target_quat,
                    output_file="outputs/new_subgoals.json"
                )
                
                # Generate new animation using existing optimizer and camera
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
                # User chose to keep original waypoints
                original_target_pose = animator.original_position
                if hasattr(animator.original_position, 'cpu'):
                    original_target_pose = animator.original_position.cpu().numpy()
                
                original_target_rotation = animator.original_rotation
                if hasattr(original_target_rotation, 'wxyz'):
                    original_target_quat = original_target_rotation.wxyz
                    if hasattr(original_target_quat, 'cpu'):
                        original_target_quat = original_target_quat.cpu().numpy()
                else:
                    # Fallback to identity quaternion if rotation format is unexpected
                    original_target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
                
                new_subgoals_file = create_new_subgoals_json(
                    original_waypoints_file=waypoints_file,
                    selected_pose_data=None,
                    original_target_pose=original_target_pose,
                    original_target_orient=original_target_quat,
                    output_file="outputs/new_subgoals.json"
                )
                
                print(f"\n=== FINAL RESULTS ===")
                print(f"New subgoals file: {new_subgoals_file} (unchanged from original)")
                print(f"No new animation generated (keeping original waypoints)")
    
    # Generate video if requested
    if generate_video:
        print(f"\nGenerating MP4 animation video...")
        
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
        
        # Ensure output directory exists
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save video
        print(f"Saving video to {output_video_path}...")
        out_clip = mpy.ImageSequenceClip(frames, fps=fps)
        out_clip.write_videofile(output_video_path, codec='libx264')
        
        # Restore original opacities
        if original_opacities is not None:
            optimizer.pipeline.model.gauss_params["opacities"] = original_opacities
        
        print(f"Video generation complete! Saved to {output_video_path}")
    
    print(f"\nProcessing complete!")
    print(f"Point cloud scene data saved to {len(saved_directories)} directories")
    if enable_random_poses and enable_pose_ranking:
        print(f"Pose generation, filtering, and ranking completed successfully")
    if generate_video:
        print(f"MP4 video saved to: {output_video_path}")


if __name__ == "__main__":
    tyro.cli(main)