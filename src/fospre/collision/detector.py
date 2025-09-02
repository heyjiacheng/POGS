"""
SDF-based collision detection between point clouds.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Dict
from scipy.spatial import cKDTree


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
        
        # Detect collision with tolerance for noise
        # Allow some collision points if they represent a small fraction (likely noise)
        collision_detected = (min_distance < self.collision_threshold and 
                            collision_ratio > 0.01)  # Allow up to 1% collision points

        collision_info = {
            "min_distance": float(min_distance),
            "collision_points": int(collision_points),
            "total_moving_points": len(moving_points),
            "collision_ratio": float(collision_ratio),
            "collision_threshold": self.collision_threshold,
            "method": "sdf"
        }
        
        return collision_detected, min_distance, collision_info