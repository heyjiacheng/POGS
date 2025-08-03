#!/usr/bin/env python3
"""
Comprehensive pose sampling and filtering script.

This script combines functionality from animate_gaussians_and_pointcloud.py and 
filter_collision_poses.py to:
1. Sample all feasible poses using pose constraints from waypoints
2. Filter out poses with collisions using SDF-based collision detection
3. Rank non-collision poses by movement amplitude relative to the last waypoint
"""

import torch
import numpy as np
import tyro
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import warp as wp
import trimesh
import open3d as o3d

from pogs.tracking.optim import Optimizer
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
from nerfstudio.cameras.cameras import Cameras


def load_camera_calibration(tf_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera position and rotation from calibration file."""
    with open(tf_file_path, 'r') as f:
        lines = f.readlines()
    
    camera_position = np.array([float(x) for x in lines[2].strip().split()])
    rotation_matrix = np.zeros((3, 3))
    for i, line in enumerate(lines[3:6]):
        rotation_matrix[i] = [float(x) for x in line.strip().split()]
    
    return camera_position, rotation_matrix


class CollisionDetector:
    """SDF-based collision detection between point clouds."""
    
    def __init__(self, collision_threshold: float = 0.005):
        self.collision_threshold = collision_threshold
    
    def compute_sdf(self, query_points: np.ndarray, reference_points: np.ndarray, 
                   k_neighbors: int = 5) -> np.ndarray:
        """Compute signed distance field values for query points relative to reference points."""
        if len(reference_points) == 0:
            return np.full(len(query_points), np.inf)
        
        tree = cKDTree(reference_points)
        distances, _ = tree.query(query_points, k=min(k_neighbors, len(reference_points)))
        
        if k_neighbors == 1 or len(reference_points) == 1:
            distances = distances.reshape(-1, 1)
        
        min_distances = distances[:, 0]
        return min_distances
    
    def detect_collision_sdf(self, moving_pcd: o3d.geometry.PointCloud, 
                            static_pcd: o3d.geometry.PointCloud) -> Tuple[bool, float, Dict]:
        """Detect collision between moving and static point clouds using SDF."""
        moving_points = np.asarray(moving_pcd.points)
        static_points = np.asarray(static_pcd.points)
        
        if len(moving_points) == 0 or len(static_points) == 0:
            return False, np.inf, {"reason": "empty_pointcloud"}
        
        sdf_values = self.compute_sdf(moving_points, static_points)
        min_distance = np.min(sdf_values)
        collision_points = np.sum(sdf_values < self.collision_threshold)
        collision_ratio = collision_points / len(moving_points)
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


class GaussianPointCloudAnimator:
    """Animates both Gaussian splats and corresponding point clouds for a target object."""
    
    def __init__(self, optimizer: Optimizer, 
                 pointcloud_path: str,
                 object_idx: int, 
                 waypoints: List[Tuple[float, float, float]], 
                 orientations: Optional[List[Tuple[float, float, float, float]]] = None):
        self.optimizer = optimizer
        self.pointcloud_path = pointcloud_path
        self.object_idx = object_idx
        self.waypoints = np.array(waypoints)
        
        self.full_pcd = o3d.io.read_point_cloud(pointcloud_path)
        if len(self.full_pcd.points) == 0:
            raise ValueError(f"Failed to load point cloud from {pointcloud_path}")
        
        if orientations is None:
            original_tf = self.optimizer.get_parts2world()[object_idx]
            self.orientations = [original_tf.rotation().wxyz] * len(waypoints)
        else:
            self.orientations = orientations
            
        self.original_pose = self.optimizer.get_parts2world()[object_idx]
        self.original_position = self.original_pose.translation()
        self.original_rotation = self.original_pose.rotation()
        
        self.original_part_deltas = self.optimizer.optimizer.part_deltas.clone()
        self.initial_part2world = self.optimizer.optimizer.get_initial_part2world(object_idx)
        
        self.target_object_pcd = self._extract_target_object_pointcloud()
        self.static_objects_pcd = self._extract_static_objects_pointcloud()
        
    def _extract_target_object_pointcloud(self) -> o3d.geometry.PointCloud:
        """Extract point cloud points that correspond to the target object using Gaussian masks."""
        group_masks = self.optimizer.optimizer.group_masks
        if self.object_idx >= len(group_masks):
            print(f"Warning: Object index {self.object_idx} out of range, using full point cloud")
            return self.full_pcd
        
        target_mask = group_masks[self.object_idx]
        gaussian_positions = self.optimizer.pipeline.model.means[target_mask].detach().cpu().numpy()
        
        full_points = np.asarray(self.full_pcd.points)
        full_colors = np.asarray(self.full_pcd.colors) if len(self.full_pcd.colors) > 0 else None
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(full_points)
        distances, indices = nbrs.kneighbors(gaussian_positions)
        
        if distances.min() < 1e-6:
            threshold = np.percentile(distances, 90)
        else:
            threshold = max(0.01, np.percentile(distances, 50))
        
        valid_indices = indices[distances.flatten() < threshold].flatten()
        valid_indices = np.unique(valid_indices)
        
        if len(valid_indices) == 0:
            threshold = np.percentile(distances, 25)
            valid_indices = indices[distances.flatten() < threshold].flatten()
            valid_indices = np.unique(valid_indices)
        
        if len(valid_indices) == 0:
            margin = 0.05
            min_bounds = gaussian_positions.min(axis=0) - margin
            max_bounds = gaussian_positions.max(axis=0) + margin
            mask = ((full_points >= min_bounds) & (full_points <= max_bounds)).all(axis=1)
            valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            target_points = full_points[valid_indices]
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            
            if full_colors is not None and len(valid_indices) <= len(full_colors):
                target_colors = full_colors[valid_indices]
                target_pcd.colors = o3d.utility.Vector3dVector(target_colors)
            
            self.original_pcd_centroid = target_points.mean(axis=0)
            return target_pcd
        else:
            raise ValueError(f"No point cloud points found for target object at index {self.object_idx}")
    
    def _extract_static_objects_pointcloud(self) -> o3d.geometry.PointCloud:
        """Extract point cloud points for all static objects (everything except target object)."""
        full_points = np.asarray(self.full_pcd.points)
        full_colors = np.asarray(self.full_pcd.colors) if len(self.full_pcd.colors) > 0 else None
        
        target_points = np.asarray(self.target_object_pcd.points)
        
        if len(target_points) == 0:
            return self.full_pcd
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
        distances, _ = nbrs.kneighbors(full_points)
        
        static_mask = distances.flatten() > 1e-6
        static_indices = np.where(static_mask)[0]
        
        static_points = full_points[static_indices]
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(static_points)
        
        if full_colors is not None and len(static_indices) <= len(full_colors):
            static_colors = full_colors[static_indices]
            static_pcd.colors = o3d.utility.Vector3dVector(static_colors)
        
        return static_pcd
    
    def update_object_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Update the pose of the specified object in the optimizer (for Gaussians)."""
        initial_position = self.initial_part2world[:3, 3].cpu().numpy()
        position_delta = position - initial_position
        
        rot_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        target_rotation = R.from_quat(rot_xyzw)
        relative_quat = target_rotation.as_quat()
        
        new_delta = torch.tensor([
            position_delta[0], position_delta[1], position_delta[2],
            relative_quat[3], relative_quat[0], relative_quat[1], relative_quat[2],
        ], dtype=torch.float32, device=self.optimizer.optimizer.part_deltas.device)
        
        self.optimizer.optimizer.part_deltas = self.original_part_deltas.clone()
        self.optimizer.optimizer.part_deltas[self.object_idx] = new_delta
    
    def generate_transformed_target_pointcloud(self, position: np.ndarray, orientation: np.ndarray) -> o3d.geometry.PointCloud:
        """Generate the actual transformed point cloud for the target object."""
        rot_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        rotation_matrix = R.from_quat(rot_xyzw).as_matrix()
        
        points = np.asarray(self.target_object_pcd.points)
        colors = np.asarray(self.target_object_pcd.colors) if len(self.target_object_pcd.colors) > 0 else None
        
        centered_points = points - self.original_pcd_centroid
        rotated_points = centered_points @ rotation_matrix.T
        final_points = rotated_points + position
        
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(final_points)
        
        if colors is not None:
            transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return transformed_pcd
    
    def render_frame(self, camera):
        """Render a single frame with the current object pose."""
        self.optimizer.optimizer.apply_to_model(
            self.optimizer.optimizer.part_deltas,
            self.optimizer.optimizer.group_labels
        )
        
        outputs = self.optimizer.pipeline.model.get_outputs(
            camera.to('cuda'),
            tracking=False,
            BLOCK_WIDTH=16,
            rgb_only=True
        )
        
        rgb_image = outputs["rgb"].squeeze().detach().cpu().numpy()
        return (rgb_image * 255).astype(np.uint8)


def sample_all_feasible_poses(base_position: np.ndarray, base_orientation: np.ndarray,
                             num_poses: int = 1000, max_translation: float = 0.05,
                             min_z: float = 0.005, max_rotation_deg: float = 180.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate random poses around a base pose with constraints."""
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
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_poses} poses")
    
    print(f"Successfully generated {len(positions)} random poses")
    return positions, orientations


def rank_poses_by_movement(positions: List[np.ndarray], orientations: List[np.ndarray],
                          reference_position: np.ndarray, reference_orientation: np.ndarray,
                          translation_weight: float = 1.0, rotation_weight: float = 0.5) -> List[int]:
    """Rank poses by movement amplitude relative to reference pose (lower movement = higher rank)."""
    print(f"Ranking {len(positions)} poses by movement amplitude...")
    
    ref_rot_xyzw = np.array([reference_orientation[1], reference_orientation[2], reference_orientation[3], reference_orientation[0]])
    ref_rotation = R.from_quat(ref_rot_xyzw)
    
    scores = []
    for pos, orient in zip(positions, orientations):
        # Translation distance
        translation_distance = np.linalg.norm(pos - reference_position)
        
        # Rotation distance (angular difference)
        curr_rot_xyzw = np.array([orient[1], orient[2], orient[3], orient[0]])
        curr_rotation = R.from_quat(curr_rot_xyzw)
        
        # Compute relative rotation
        relative_rotation = ref_rotation.inv() * curr_rotation
        rotation_angle = np.abs(relative_rotation.as_rotvec())
        rotation_distance = np.linalg.norm(rotation_angle)
        
        # Combined score (lower is better)
        score = translation_weight * translation_distance + rotation_weight * rotation_distance
        scores.append(score)
    
    # Sort indices by score (ascending - lower movement first)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i])
    
    print(f"Movement scores: min={min(scores):.4f}, max={max(scores):.4f}, avg={np.mean(scores):.4f}")
    return ranked_indices


def save_pose_data(animator: GaussianPointCloudAnimator, camera, position: np.ndarray,
                   orientation: np.ndarray, pose_idx: int, rank: int, score: float,
                   output_base_dir: str = "outputs", timestamp: str = None) -> Path:
    """Save both Gaussian render and point cloud data for a specific pose."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    pose_dir = Path(output_base_dir) / f"filtered_poses_{timestamp}" / f"pose_{pose_idx:03d}_rank_{rank:03d}"
    pose_dir.mkdir(parents=True, exist_ok=True)
    
    animator.update_object_pose(position, orientation)
    gaussian_image = animator.render_frame(camera)
    
    gaussian_path = pose_dir / "gaussian_render.png"
    from PIL import Image
    gaussian_pil = Image.fromarray(gaussian_image)
    gaussian_pil.save(gaussian_path)
    
    transformed_target_pcd = animator.generate_transformed_target_pointcloud(position, orientation)
    
    target_pcd_path = pose_dir / "moving_object.ply"
    static_pcd_path = pose_dir / "static_objects.ply"
    combined_pcd_path = pose_dir / "complete_scene.ply"
    
    o3d.io.write_point_cloud(str(target_pcd_path), transformed_target_pcd)
    o3d.io.write_point_cloud(str(static_pcd_path), animator.static_objects_pcd)
    
    combined_pcd = o3d.geometry.PointCloud()
    transformed_points = np.asarray(transformed_target_pcd.points)
    static_points = np.asarray(animator.static_objects_pcd.points)
    all_points = np.vstack([transformed_points, static_points])
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    if len(transformed_target_pcd.colors) > 0 and len(animator.static_objects_pcd.colors) > 0:
        transformed_colors = np.asarray(transformed_target_pcd.colors)
        static_colors = np.asarray(animator.static_objects_pcd.colors)
        all_colors = np.vstack([transformed_colors, static_colors])
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    o3d.io.write_point_cloud(str(combined_pcd_path), combined_pcd)
    
    metadata = {
        "pose_index": pose_idx,
        "rank": rank,
        "movement_score": float(score),
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


def setup_optimizer(config_path: Path) -> Optimizer:
    """Initialize the POGS optimizer."""
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
    time.sleep(2)
    
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
    camera_position, rotation_matrix = load_camera_calibration(calibration_file)
    
    cam2world = np.eye(4)
    cam2world[:3, :3] = rotation_matrix.T
    cam2world[:3, 3] = camera_position
    
    opengl_tf = cam2world @ trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    
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
    
    camera = Cameras(
        camera_to_worlds=torch.from_numpy(opengl_tf[:3, :]).float()[None, :],
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
    )
    
    camera.camera_to_worlds[:, :3, 3] *= optimizer.dataset_scale
    return camera


def load_waypoints_from_json(waypoints_file: str) -> List[Tuple[float, float, float]]:
    """Load waypoints from JSON file."""
    with open(waypoints_file, 'r') as f:
        data = json.load(f)
    
    waypoints = []
    for subgoal in data['subgoals']:
        subgoal_pose = subgoal['subgoal_pose']
        waypoints.append((subgoal_pose[0], subgoal_pose[1], subgoal_pose[2]))
    
    return waypoints


def main(
    config_path: Path = Path("outputs/box/pogs/2025-07-29_151651/config.yml"),
    pointcloud_path: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/box/prime_seg_gaussians.ply",
    semantic_query: str = "box cutter",
    waypoints_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/outputs/all_subgoals.json",
    calibration_file: str = "/home/jiachengxu/workspace/master_thesis/POGS/src/pogs/calibration_outputs/world_to_d405.tf",
    num_sample_poses: int = 1000,
    max_translation: float = 0.05,
    max_rotation_deg: float = 90.0,
    min_z: float = 0.005,
    collision_threshold: float = 0.01,
    max_output_poses: int = 50,
    translation_weight: float = 0.5,
    rotation_weight: float = 1.0,
):
    """Sample all feasible poses, filter collisions, and rank by movement amplitude.
    
    Args:
        config_path: Path to POGS config file
        pointcloud_path: Path to the PLY point cloud file
        semantic_query: Semantic description of target object
        waypoints_file: Path to JSON file containing waypoints
        calibration_file: Path to camera calibration file
        num_sample_poses: Number of poses to sample
        max_translation: Maximum translation per axis in meters
        max_rotation_deg: Maximum rotation per axis in degrees
        min_z: Minimum z-coordinate (above table surface)
        collision_threshold: Collision detection threshold in meters
        max_output_poses: Maximum number of final poses to output
        translation_weight: Weight for translation in ranking
        rotation_weight: Weight for rotation in ranking
    """
    wp.init()
    print("=== POSE SAMPLING AND FILTERING ===")
    print("Loading model...")
    
    waypoints = load_waypoints_from_json(waypoints_file)
    print(f"Loaded {len(waypoints)} waypoints from {waypoints_file}")
    
    optimizer = setup_optimizer(config_path)
    target_object_idx = find_target_object(optimizer, semantic_query)
    
    animator = GaussianPointCloudAnimator(
        optimizer=optimizer,
        pointcloud_path=pointcloud_path,
        object_idx=target_object_idx,
        waypoints=waypoints,
        orientations=None
    )
    
    final_waypoint = waypoints[-1]
    base_position = np.array(final_waypoint)
    base_orientation = animator.orientations[-1]
    
    print(f"\nUsing final waypoint as reference:")
    print(f"Reference position: [{base_position[0]:.6f}, {base_position[1]:.6f}, {base_position[2]:.6f}]")
    print(f"Reference orientation: [{base_orientation[0]:.6f}, {base_orientation[1]:.6f}, {base_orientation[2]:.6f}, {base_orientation[3]:.6f}]")
    
    # Sample all feasible poses
    print(f"\n=== STEP 1: SAMPLING FEASIBLE POSES ===")
    sample_positions, sample_orientations = sample_all_feasible_poses(
        base_position=base_position,
        base_orientation=base_orientation,
        num_poses=num_sample_poses,
        max_translation=max_translation,
        min_z=min_z,
        max_rotation_deg=max_rotation_deg
    )
    
    # Filter out collisions
    print(f"\n=== STEP 2: FILTERING COLLISIONS ===")
    collision_detector = CollisionDetector(collision_threshold)
    valid_positions = []
    valid_orientations = []
    collision_count = 0
    
    for i, (pos, orient) in enumerate(zip(sample_positions, sample_orientations)):
        # Generate transformed point cloud for collision detection
        transformed_pcd = animator.generate_transformed_target_pointcloud(pos, orient)
        
        # Check collision
        collision_detected, min_distance, collision_info = collision_detector.detect_collision_sdf(
            transformed_pcd, animator.static_objects_pcd
        )
        
        if not collision_detected:
            valid_positions.append(pos)
            valid_orientations.append(orient)
        else:
            collision_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_positions)} poses (Valid: {len(valid_positions)}, Collisions: {collision_count})")
    
    print(f"Collision filtering complete: {len(valid_positions)} valid poses, {collision_count} collisions")
    
    if len(valid_positions) == 0:
        print("ERROR: No valid poses found after collision filtering!")
        print("Try increasing collision_threshold or max_translation parameters.")
        return
    
    # Rank poses by movement amplitude
    print(f"\n=== STEP 3: RANKING POSES BY MOVEMENT ===")
    ranked_indices = rank_poses_by_movement(
        valid_positions, valid_orientations,
        base_position, base_orientation,
        translation_weight, rotation_weight
    )
    
    # Limit output poses
    max_poses = min(max_output_poses, len(valid_positions))
    final_indices = ranked_indices[:max_poses]
    
    print(f"Selected top {max_poses} poses for output")
    
    # Save results
    print(f"\n=== STEP 4: SAVING RESULTS ===")
    camera = create_camera_from_calibration(calibration_file, optimizer)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    saved_dirs = []
    movement_scores = []
    
    for i, idx in enumerate(final_indices):
        pos = valid_positions[idx]
        orient = valid_orientations[idx]
        
        # Calculate movement score for this pose
        translation_distance = np.linalg.norm(pos - base_position)
        ref_rot_xyzw = np.array([base_orientation[1], base_orientation[2], base_orientation[3], base_orientation[0]])
        curr_rot_xyzw = np.array([orient[1], orient[2], orient[3], orient[0]])
        ref_rotation = R.from_quat(ref_rot_xyzw)
        curr_rotation = R.from_quat(curr_rot_xyzw)
        relative_rotation = ref_rotation.inv() * curr_rotation
        rotation_distance = np.linalg.norm(relative_rotation.as_rotvec())
        score = translation_weight * translation_distance + rotation_weight * rotation_distance
        movement_scores.append(score)
        
        pose_dir = save_pose_data(
            animator=animator,
            camera=camera,
            position=pos,
            orientation=orient,
            pose_idx=i,
            rank=i+1,
            score=score,
            output_base_dir="outputs",
            timestamp=timestamp
        )
        saved_dirs.append(pose_dir)
        
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{max_poses} poses")
    
    # Reset to original pose
    animator.optimizer.optimizer.part_deltas = animator.original_part_deltas.clone()
    
    # Save summary
    summary = {
        "total_sampled_poses": len(sample_positions),
        "valid_poses_after_filtering": len(valid_positions),
        "collision_poses": collision_count,
        "final_output_poses": max_poses,
        "valid_ratio": len(valid_positions) / len(sample_positions),
        "parameters": {
            "max_translation": max_translation,
            "max_rotation_deg": max_rotation_deg,
            "min_z": min_z,
            "collision_threshold": collision_threshold,
            "translation_weight": translation_weight,
            "rotation_weight": rotation_weight
        },
        "reference_pose": {
            "position": base_position.tolist(),
            "orientation_wxyz": base_orientation.tolist()
        },
        "movement_score_stats": {
            "min": float(min(movement_scores)),
            "max": float(max(movement_scores)),
            "mean": float(np.mean(movement_scores)),
            "median": float(np.median(movement_scores))
        },
        "timestamp": timestamp
    }
    
    summary_path = Path("outputs") / f"filtered_poses_{timestamp}" / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== COMPLETION SUMMARY ===")
    print(f"Total poses sampled: {len(sample_positions)}")
    print(f"Valid poses (no collision): {len(valid_positions)}")
    print(f"Collision poses filtered: {collision_count}")
    print(f"Final poses output: {max_poses}")
    print(f"Valid ratio: {len(valid_positions) / len(sample_positions):.2%}")
    print(f"Movement scores: min={min(movement_scores):.4f}, max={max(movement_scores):.4f}")
    print(f"Results saved to: outputs/filtered_poses_{timestamp}/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    tyro.cli(main)