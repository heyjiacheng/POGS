"""
Gaussian splat and point cloud animation for target objects.
"""

import torch
import numpy as np
import json
import open3d as o3d
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R, Slerp

from pogs.tracking.optim import Optimizer


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
        
        # Load and filter point cloud
        self.full_pcd = self._load_and_filter_pointcloud(pointcloud_path)
        
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
        
        # Extract target object and static objects point clouds
        self.target_object_pcd = self._extract_target_object_pointcloud()
        self.static_objects_pcd = self._extract_static_objects_pointcloud()
    
    def _load_and_filter_pointcloud(self, pointcloud_path: str) -> o3d.geometry.PointCloud:
        """Load and filter point cloud file."""
        full_pcd = o3d.io.read_point_cloud(pointcloud_path)
        if len(full_pcd.points) == 0:
            raise ValueError(f"Failed to load point cloud from {pointcloud_path}")
        
        # Filter out points with z < 0.001
        full_points = np.asarray(full_pcd.points)
        full_colors = np.asarray(full_pcd.colors) if len(full_pcd.colors) > 0 else None
        
        z_mask = full_points[:, 2] >= 0.001
        filtered_points = full_points[z_mask]
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        if full_colors is not None:
            filtered_colors = full_colors[z_mask]
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        print(f"Loaded point cloud with {len(filtered_points)} points (filtered out {len(full_points) - len(filtered_points)} points with z < 0.001)")
        return filtered_pcd
    
    def _remove_outliers(self, pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
        """Remove outliers from point cloud using statistical outlier removal."""
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
        
        # Find point cloud points that are close to target Gaussians
        full_points = np.asarray(self.full_pcd.points)
        full_colors = np.asarray(self.full_pcd.colors) if len(self.full_pcd.colors) > 0 else None
        
        # Use KD-tree to find nearest point cloud points to Gaussians
        from sklearn.neighbors import NearestNeighbors
        
        # Find point cloud points within a threshold distance of any Gaussian
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(full_points)
        distances, indices = nbrs.kneighbors(gaussian_positions)
        
        print(f"Distance statistics: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}")
        
        # Use adaptive threshold
        if distances.min() < 1e-6:  # Very close matches exist
            threshold = np.percentile(distances, 90)  # Use 90th percentile
        else:
            threshold = max(0.01, np.percentile(distances, 50))  # At least 1cm or median distance
        
        valid_indices = indices[distances.flatten() < threshold].flatten()
        valid_indices = np.unique(valid_indices)  # Remove duplicates
        
        print(f"Extracted {len(valid_indices)} point cloud points for target object (threshold: {threshold:.6f})")
        
        # If no points found, try spatial proximity approach
        if len(valid_indices) == 0:
            print("No points found with distance-based approach, trying spatial proximity...")
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
            
            # Calculate and store original centroid
            target_points = np.asarray(target_pcd.points)
            self.original_pcd_centroid = target_points.mean(axis=0)
            print(f"Original target object point cloud centroid: {self.original_pcd_centroid}")
            
            return target_pcd
        else:
            raise ValueError(f"No point cloud points found for target object at index {self.object_idx}")
    
    def _extract_static_objects_pointcloud(self) -> o3d.geometry.PointCloud:
        """Extract point cloud points for all static objects (everything except target object)."""
        # Get all point cloud points
        full_points = np.asarray(self.full_pcd.points)
        full_colors = np.asarray(self.full_pcd.colors) if len(self.full_pcd.colors) > 0 else None
        
        # Get target object points to exclude them
        target_points = np.asarray(self.target_object_pcd.points)
        
        if len(target_points) == 0:
            return self.full_pcd
        
        # Find indices of points that are NOT part of the target object
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