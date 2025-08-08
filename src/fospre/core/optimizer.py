"""
POGS optimizer setup and management.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from pogs.tracking.optim import Optimizer
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
from .utils import safe_sleep


def setup_optimizer(config_path: Path) -> Optimizer:
    """Initialize the POGS optimizer.
    
    Args:
        config_path: Path to the POGS config file
        
    Returns:
        Initialized Optimizer instance
    """
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
    safe_sleep(2)  # Wait for optimizer to load
    
    return optimizer


def cluster_point_cloud(points: np.ndarray, eps: float = 0.02, min_samples: int = 10) -> List[np.ndarray]:
    """Cluster point cloud points using DBSCAN.
    
    Args:
        points: Point cloud points (N x 3)
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        List of point clusters
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    clusters = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
        print(f"Cluster {label}: {len(cluster_points)} points")
    
    return clusters


def find_target_object_by_subgoals(optimizer: Optimizer, waypoints_file: str, pointcloud_path: str = None) -> int:
    """Find target object based on proximity to grasp stage subgoal poses.
    
    Args:
        optimizer: POGS optimizer instance
        waypoints_file: Path to all_subgoals.json file
        pointcloud_path: Optional path to point cloud for clustering
        
    Returns:
        Index of the target object
    """
    # Load subgoals
    with open(waypoints_file, 'r') as f:
        subgoals_data = json.load(f)
    
    # Find grasp stage subgoals
    grasp_poses = []
    for subgoal in subgoals_data['subgoals']:
        if subgoal.get('is_grasp_stage', False):
            pose = subgoal['subgoal_pose']
            grasp_poses.append(np.array(pose[:3]))  # Extract position (x,y,z)
    
    if not grasp_poses:
        raise ValueError("No grasp stage subgoals found in waypoints file")
    
    print(f"Found {len(grasp_poses)} grasp stage subgoal(s)")
    
    # Get all object masks and their Gaussian positions
    group_masks = optimizer.optimizer.group_masks
    gaussian_means = optimizer.pipeline.model.means.detach().cpu().numpy()
    
    min_distance = float('inf')
    target_idx = 0
    
    # For each object, calculate minimum distance to any grasp pose
    for obj_idx, mask in enumerate(group_masks):
        if torch.sum(mask) == 0:  # Skip empty masks
            continue
            
        # Get Gaussian positions for this object
        obj_gaussians = gaussian_means[mask.cpu().numpy()]
        
        if len(obj_gaussians) == 0:
            continue
        
        # Calculate centroid of this object's Gaussians
        obj_centroid = obj_gaussians.mean(axis=0)
        
        # Find minimum distance to any grasp pose
        distances = [np.linalg.norm(obj_centroid - grasp_pose) for grasp_pose in grasp_poses]
        min_obj_distance = min(distances)
        
        print(f"Object {obj_idx}: centroid {obj_centroid}, min distance to grasp pose: {min_obj_distance:.6f}")
        
        if min_obj_distance < min_distance:
            min_distance = min_obj_distance
            target_idx = obj_idx
    
    print(f"Target object (closest to grasp pose): index {target_idx} (distance: {min_distance:.6f})")
    return target_idx


def find_target_object(optimizer: Optimizer, semantic_query: str) -> int:
    """Find object index matching the semantic query (legacy CLIP-based method).
    
    Args:
        optimizer: POGS optimizer instance
        semantic_query: Semantic description of the target object
        
    Returns:
        Index of the target object
    """
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