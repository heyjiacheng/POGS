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


def find_target_objects_by_subgoals(optimizer: Optimizer, waypoints_file: str, pointcloud_path: str = None) -> List[Tuple[int, int]]:
    """Find target objects and create grasp-release stage pairs.
    
    Args:
        optimizer: POGS optimizer instance
        waypoints_file: Path to all_subgoals.json file
        pointcloud_path: Optional path to point cloud for clustering
        
    Returns:
        List of tuples (grasp_stage, object_idx) for each grasp-release pair in order
    """
    # Load subgoals
    with open(waypoints_file, 'r') as f:
        subgoals_data = json.load(f)
    
    # Find grasp stage subgoals and build grasp-release pairs
    grasp_release_pairs = []
    for subgoal in subgoals_data['subgoals']:
        if subgoal.get('is_grasp_stage', False):
            grasp_stage = subgoal['stage']
            grasp_pose = np.array(subgoal['subgoal_pose'][:3])
            
            # Find corresponding release stage (should be next stage)
            release_stage = None
            release_pose = None
            for release_subgoal in subgoals_data['subgoals']:
                if (release_subgoal['stage'] == grasp_stage + 1 and 
                    release_subgoal.get('is_release_stage', False)):
                    release_stage = release_subgoal['stage']
                    release_pose = np.array(release_subgoal['subgoal_pose'][:3])
                    break
            
            if release_stage is not None:
                grasp_release_pairs.append((grasp_stage, release_stage, grasp_pose, release_pose))
                print(f"Grasp-Release pair: Stage {grasp_stage} (grasp) -> Stage {release_stage} (release)")
            else:
                print(f"Warning: No release stage found for grasp stage {grasp_stage}")
    
    if not grasp_release_pairs:
        raise ValueError("No grasp-release pairs found in waypoints file")
    
    # Sort by grasp stage number to maintain order
    grasp_release_pairs.sort(key=lambda x: x[0])
    print(f"Found {len(grasp_release_pairs)} grasp-release pair(s)")
    
    # Get all object masks and their Gaussian positions
    group_masks = optimizer.optimizer.group_masks
    gaussian_means = optimizer.pipeline.model.means.detach().cpu().numpy()
    
    # Calculate object centroids once
    object_centroids = {}
    for obj_idx, mask in enumerate(group_masks):
        if torch.sum(mask) == 0:  # Skip empty masks
            continue
            
        # Get Gaussian positions for this object
        obj_gaussians = gaussian_means[mask.cpu().numpy()]
        
        if len(obj_gaussians) == 0:
            continue
        
        # Calculate centroid of this object's Gaussians
        obj_centroid = obj_gaussians.mean(axis=0)
        object_centroids[obj_idx] = obj_centroid
    
    # Find closest object for each grasp stage in order
    target_objects = []
    used_objects = set()  # Track which objects have been assigned
    
    for grasp_stage, release_stage, grasp_pose, release_pose in grasp_release_pairs:
        min_distance = float('inf')
        target_idx = None
        
        # Find the closest unused object to this grasp pose
        for obj_idx, obj_centroid in object_centroids.items():
            if obj_idx in used_objects:  # Skip already assigned objects
                continue
                
            distance = np.linalg.norm(obj_centroid - grasp_pose)
            print(f"Grasp Stage {grasp_stage}, Object {obj_idx}: distance to grasp pose: {distance:.6f}")
            
            if distance < min_distance:
                min_distance = distance
                target_idx = obj_idx
        
        if target_idx is not None:
            # Return grasp stage and object index for this pair
            target_objects.append((grasp_stage, target_idx))
            used_objects.add(target_idx)
            print(f"Grasp Stage {grasp_stage}: assigned to object {target_idx} (distance: {min_distance:.6f})")
        else:
            print(f"Warning: No available object found for grasp stage {grasp_stage}")
    
    return target_objects


def find_target_object_by_subgoals(optimizer: Optimizer, waypoints_file: str, pointcloud_path: str = None) -> int:
    """Find target object based on proximity to grasp stage subgoal poses (legacy single-object method).
    
    Args:
        optimizer: POGS optimizer instance
        waypoints_file: Path to all_subgoals.json file
        pointcloud_path: Optional path to point cloud for clustering
        
    Returns:
        Index of the target object
    """
    target_objects = find_target_objects_by_subgoals(optimizer, waypoints_file, pointcloud_path)
    if target_objects:
        return target_objects[0][1]  # Return first object index
    else:
        raise ValueError("No target objects found")


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