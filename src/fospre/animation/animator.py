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
        self.ds = self.optimizer.optimizer.dataset_scale

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
        gaussian_positions = (self.optimizer.pipeline.model.means[target_mask]
                      .detach().cpu().numpy() / self.ds)

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
        position_delta = position * self.ds - initial_position

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
        
        # Update only the specific object's delta while preserving others
        current_deltas = self.optimizer.optimizer.part_deltas.detach().clone()
        current_deltas[self.object_idx] = new_delta
        self.optimizer.optimizer.part_deltas = current_deltas
    
    def generate_transformed_target_pointcloud(self, position: np.ndarray, orientation: np.ndarray) -> o3d.geometry.PointCloud:
        """Generate the actual transformed point cloud for the target object."""
        # Convert orientation to rotation matrix
        rot_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        rotation_matrix = R.from_quat(rot_xyzw).as_matrix()
        
        # Get target object points
        points = np.asarray(self.target_object_pcd.points)
        colors = np.asarray(self.target_object_pcd.colors) if len(self.target_object_pcd.colors) > 0 else None
        
        reference_position = np.asarray(self.original_pcd_centroid)

        # Center the point cloud at origin (relative to initial Gaussian position)
        centered_points = points - reference_position
        
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


class MultiStageAnimator:
    """Animator for handling multiple objects moving in sequence based on subgoal stages."""
    
    def __init__(self, optimizer: Optimizer, pointcloud_path: str, target_objects: List[Tuple[int, int]], 
                 waypoints_file: str):
        """Initialize multi-stage animator.
        
        Args:
            optimizer: POGS optimizer containing the scene
            pointcloud_path: Path to the PLY point cloud file
            target_objects: List of (stage, object_idx) tuples in order
            waypoints_file: Path to all_subgoals.json file
        """
        self.optimizer = optimizer
        self.pointcloud_path = pointcloud_path
        self.target_objects = target_objects
        self.waypoints_file = waypoints_file
        self.ds = self.optimizer.optimizer.dataset_scale
        print(f"Dataset scale: {self.ds}")
        
        # Load subgoals data
        with open(waypoints_file, 'r') as f:
            self.subgoals_data = json.load(f)
        
        # Store original part deltas
        self.original_part_deltas = self.optimizer.optimizer.part_deltas.clone()
        
        # Create individual animators for each stage
        self.stage_animators = {}
        self._create_stage_animators()
    
    def _convert_ee_to_target_waypoints(self, subgoals, target_obj_idx):
        """Convert end effector waypoints to target object waypoints.
        
        Args:
            subgoals: List of subgoal dictionaries with subgoal_pose
            target_obj_idx: Index of target object
            
        Returns:
            Tuple of (waypoints, orientations) for target object
        """
        # Get original target object pose from initial Gaussians (before any transformations)
        # Use initial part2world to get the object's original position, not current position
        initial_part2world = self.optimizer.optimizer.get_initial_part2world(target_obj_idx)
        original_position = (initial_part2world[:3, 3] / self.ds if hasattr(initial_part2world, '__array__')
                     else np.array(initial_part2world) / self.ds)

        # Handle different types of position data
        if hasattr(original_position, 'cpu'):
            original_position = original_position.cpu().numpy()
        elif not isinstance(original_position, np.ndarray):
            original_position = np.array(original_position)
        
        # Get original rotation from initial part2world (assuming identity rotation at start)
        # Since initial_part2world represents the original pose, rotation should be identity
        original_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] - identity quaternion
        
        print(f"Original target object pose: [{original_position[0]:.6f}, {original_position[1]:.6f}, {original_position[2]:.6f}]")
        print(f"Original target object orient: [{original_quat[0]:.6f}, {original_quat[1]:.6f}, {original_quat[2]:.6f}, {original_quat[3]:.6f}]")
        
        # Get the first end effector pose (grasp pose) to calculate relative transform
        first_subgoal = subgoals[0]['subgoal_pose']
        first_ee_pos = np.array(first_subgoal[:3])
        # Convert from [x, y, z, qx, qy, qz, qw] to [w, x, y, z] format
        first_ee_orient = np.array([first_subgoal[6], first_subgoal[3], first_subgoal[4], first_subgoal[5]])
        
        # Import transform functions
        from ..pose.transforms import get_ee_target_relative_transform, apply_relative_transform
        
        # Calculate the relative transform from first end effector to target object
        # During grasp, ee and target should be in the same relative position
        relative_pos, relative_orient = get_ee_target_relative_transform(
            first_ee_pos, first_ee_orient,
            original_position, original_quat
        )
        
        print(f"Calculated relative transform from EE to target:")
        print(f"  Relative position: [{relative_pos[0]:.6f}, {relative_pos[1]:.6f}, {relative_pos[2]:.6f}]")
        print(f"  Relative orientation: [{relative_orient[0]:.6f}, {relative_orient[1]:.6f}, {relative_orient[2]:.6f}, {relative_orient[3]:.6f}]")
        
        # Apply the same relative transform to all end effector waypoints
        target_waypoints = []
        target_orientations = []
        
        for subgoal in subgoals:
            subgoal_pose = subgoal['subgoal_pose']
            ee_pos = np.array(subgoal_pose[:3])  # [x, y, z]
            # Convert from [x, y, z, qx, qy, qz, qw] to [w, x, y, z] format
            ee_orient = np.array([subgoal_pose[6], subgoal_pose[3], subgoal_pose[4], subgoal_pose[5]])
            
            # Apply relative transform to get target object pose
            target_pos, target_orient = apply_relative_transform(
                ee_pos, ee_orient,
                relative_pos, relative_orient
            )
            
            target_waypoints.append(target_pos)
            target_orientations.append(target_orient)
        
        print(f"Target waypoint conversion result:")
        print(f"  First target pose:  [{target_waypoints[0][0]:.6f}, {target_waypoints[0][1]:.6f}, {target_waypoints[0][2]:.6f}]")
        print(f"  Final target pose:  [{target_waypoints[-1][0]:.6f}, {target_waypoints[-1][1]:.6f}, {target_waypoints[-1][2]:.6f}]")
        
        return target_waypoints, target_orientations
    
    
    def _create_stage_animators(self):
        """Create individual animators for each grasp-release pair."""
        # Group subgoals by stage
        stage_subgoals = {}
        for subgoal in self.subgoals_data['subgoals']:
            stage = subgoal['stage']
            stage_subgoals[stage] = subgoal
        
        # Create animators for grasp-release pairs
        for grasp_stage, obj_idx in self.target_objects:
            # Find corresponding release stage (should be next stage)
            release_stage = grasp_stage + 1
            
            if grasp_stage not in stage_subgoals:
                print(f"Warning: No grasp subgoal found for stage {grasp_stage}")
                continue
                
            if release_stage not in stage_subgoals:
                print(f"Warning: No release subgoal found for stage {release_stage}")
                continue
            
            # Get waypoints: grasp position -> release position
            grasp_goal = stage_subgoals[grasp_stage]
            release_goal = stage_subgoals[release_stage]
            
            # Convert ee waypoints to target object waypoints
            target_waypoints, target_orientations = self._convert_ee_to_target_waypoints(
                [grasp_goal, release_goal], obj_idx
            )
            
            waypoints = target_waypoints
            orientations = target_orientations
            
            print(f"Grasp-Release pair {grasp_stage}-{release_stage}: Creating animator for object {obj_idx}")
            print(f"  Grasp position: {waypoints[0]}")
            print(f"  Release position: {waypoints[1]}")
            
            # Create single-object animator for this grasp-release pair
            animator = GaussianPointCloudAnimator(
                optimizer=self.optimizer,
                pointcloud_path=self.pointcloud_path,
                object_idx=obj_idx,
                waypoints=waypoints,
                orientations=orientations
            )
            
            # Use grasp stage as key for the animator
            self.stage_animators[grasp_stage] = animator
    
    def generate_sequential_animation(self, camera, duration_per_stage: float = 5.0, fps: int = 30):
        """Generate animation with objects moving in sequence.
        
        Args:
            camera: Camera for rendering
            duration_per_stage: Duration for each stage in seconds
            fps: Frames per second
            
        Returns:
            List of rendered frames
        """
        all_frames = []
        
        print(f"Generating sequential animation for {len(self.target_objects)} grasp-release pairs")
        
        # Track cumulative object positions for sequential movement
        cumulative_positions = {}
        
        for stage_idx, (grasp_stage, obj_idx) in enumerate(self.target_objects):
            print(f"\\nProcessing Grasp-Release pair {grasp_stage} (Object {obj_idx})...")
            
            if grasp_stage not in self.stage_animators:
                print(f"Warning: No animator for grasp stage {grasp_stage}")
                continue
            
            animator = self.stage_animators[grasp_stage]
            
            print(f"  Grasp position: {animator.waypoints[0]}")
            print(f"  Release position: {animator.waypoints[1]}")
            
            # Start from original state 
            current_part_deltas = self.original_part_deltas.clone()
            
            # Apply cumulative transformations for all previous objects at their final positions
            for prev_obj_idx, (final_pos, final_orient) in cumulative_positions.items():
                if prev_obj_idx != obj_idx:  # Don't apply to current object
                    # Calculate delta for previous object's final position
                    initial_part2world = self.optimizer.optimizer.get_initial_part2world(prev_obj_idx)
                    initial_position = initial_part2world[:3, 3].cpu().numpy()
                    position_delta = final_pos * self.ds - initial_position
                    
                    # Convert orientation to rotation matrix
                    rot_xyzw = np.array([final_orient[1], final_orient[2], final_orient[3], final_orient[0]])
                    target_rotation = R.from_quat(rot_xyzw)
                    relative_quat = target_rotation.as_quat()  # xyzw format
                    
                    # Create delta tensor for previous object
                    prev_delta = torch.tensor([
                        position_delta[0],
                        position_delta[1], 
                        position_delta[2],
                        relative_quat[3],  # w
                        relative_quat[0],  # x
                        relative_quat[1],  # y
                        relative_quat[2],  # z
                    ], dtype=torch.float32, device=current_part_deltas.device)
                    
                    # Apply delta to current state
                    current_part_deltas[prev_obj_idx] = prev_delta
                    print(f"  Applied final position to previous object {prev_obj_idx}: {final_pos}")
            
            # Set the accumulated deltas (detach to avoid gradient issues)
            self.optimizer.optimizer.part_deltas = current_part_deltas.detach()
            
            # Generate frames for this grasp-release pair
            stage_frames = animator.animate(camera, duration_per_stage, fps)
            all_frames.extend(stage_frames)
            
            # Store final position and orientation for this object
            final_waypoint = animator.waypoints[-1]  # Release position
            final_orientation = animator.orientations[-1]  # Release orientation
            cumulative_positions[obj_idx] = (final_waypoint, final_orientation)
            
            print(f"Grasp-Release pair {grasp_stage} complete: {len(stage_frames)} frames")
            print(f"Final position stored for object {obj_idx}: {final_waypoint}")
        
        # Reset to original state
        self.optimizer.optimizer.part_deltas = self.original_part_deltas.clone()
        
        print(f"Total animation: {len(all_frames)} frames")
        return all_frames
    
    def save_stage_data(self, output_dir: str):
        """Generate and save scene data for all stages.
        
        Args:
            output_dir: Directory to save stage data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stage_info = {
            'total_stages': len(self.target_objects),
            'stages': []
        }
        
        all_saved_dirs = []
        
        # Generate scene animation data for each grasp-release pair
        for stage_idx, (grasp_stage, obj_idx) in enumerate(self.target_objects, 1):
            if grasp_stage not in self.stage_animators:
                continue
                
            animator = self.stage_animators[grasp_stage]
            release_stage = grasp_stage + 1
            stage_output_dir = output_path / f"stage_{stage_idx}"
            
            print(f"\nGenerating scene animation data for Stage {stage_idx} (Grasp {grasp_stage}-Release {release_stage}, Object {obj_idx})...")
            print(f"  Grasp position: {animator.waypoints[0]}")
            print(f"  Release position: {animator.waypoints[1]}")
            
            # Import generate_scene_animation function
            import sys
            sys.path.append('/home/jiachengxu/workspace/master_thesis/POGS/scripts')
            from sample_and_filter_poses import generate_scene_animation
            
            # Generate scene animation for this grasp-release pair
            saved_dirs = generate_scene_animation(animator, f"stage_{stage_idx}_obj_{obj_idx}", str(stage_output_dir))
            all_saved_dirs.extend(saved_dirs)
            
            stage_data = {
                'stage_number': stage_idx,
                'grasp_stage': grasp_stage,
                'release_stage': release_stage,
                'object_idx': obj_idx,
                'grasp_position': animator.waypoints[0].tolist() if hasattr(animator.waypoints[0], 'tolist') else list(animator.waypoints[0]),
                'release_position': animator.waypoints[1].tolist() if hasattr(animator.waypoints[1], 'tolist') else list(animator.waypoints[1]),
                'grasp_orientation': list(animator.orientations[0]) if not isinstance(animator.orientations[0], list) else animator.orientations[0],
                'release_orientation': list(animator.orientations[1]) if not isinstance(animator.orientations[1], list) else animator.orientations[1],
                'scene_directories': [str(d) for d in saved_dirs]  # Convert Path objects to strings
            }
            stage_info['stages'].append(stage_data)
        
        # Save stage info
        with open(output_path / 'multi_stage_info.json', 'w') as f:
            json.dump(stage_info, f, indent=2)
        
        print(f"\nSaved multi-stage info to {output_path / 'multi_stage_info.json'}")
        print(f"Total scene directories: {len(all_saved_dirs)}")
        
        return str(output_path)