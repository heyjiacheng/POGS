#!/usr/bin/env python3
"""
Collision detection and pose filtering script.

This script filters out poses where the moving object collides with static objects
using SDF-based collision detection methods.
"""

import numpy as np
import open3d as o3d
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from datetime import datetime
from scipy.spatial import cKDTree
import trimesh


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
        distances, indices = tree.query(query_points, k=min(k_neighbors, len(reference_points)))
        
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
    
    def detect_collision_mesh(self, moving_pcd: o3d.geometry.PointCloud, 
                             static_pcd: o3d.geometry.PointCloud) -> Tuple[bool, float, Dict]:
        """
        Alternative collision detection using mesh-based approach.
        
        Args:
            moving_pcd: Moving object point cloud
            static_pcd: Static objects point cloud
            
        Returns:
            Tuple of (collision_detected, min_distance, collision_info)
        """
        try:
            moving_points = np.asarray(moving_pcd.points)
            static_points = np.asarray(static_pcd.points)
            
            if len(moving_points) == 0 or len(static_points) == 0:
                return False, np.inf, {"reason": "empty_pointcloud"}
            
            # Create convex hulls for both point clouds
            static_hull = o3d.geometry.PointCloud()
            static_hull.points = o3d.utility.Vector3dVector(static_points)
            static_convex_hull, _ = static_hull.compute_convex_hull()
            
            # Check if moving points are inside the static convex hull
            inside_count = 0
            min_distance = np.inf
            
            for point in moving_points:
                # Simple distance-based check to convex hull vertices
                hull_vertices = np.asarray(static_convex_hull.vertices)
                distances = np.linalg.norm(hull_vertices - point, axis=1)
                min_dist_to_hull = np.min(distances)
                min_distance = min(min_distance, min_dist_to_hull)
                
                if min_dist_to_hull < self.collision_threshold:
                    inside_count += 1
            
            collision_detected = inside_count > 0
            collision_ratio = inside_count / len(moving_points)
            
            collision_info = {
                "min_distance": float(min_distance),
                "collision_points": int(inside_count),
                "total_moving_points": len(moving_points),
                "collision_ratio": float(collision_ratio),
                "collision_threshold": self.collision_threshold,
                "method": "convex_hull"
            }
            
            return collision_detected, min_distance, collision_info
            
        except Exception as e:
            # Fallback to SDF method
            return self.detect_collision_sdf(moving_pcd, static_pcd)


class PoseFilter:
    """Filter poses based on collision detection."""
    
    def __init__(self, input_dir: str, collision_threshold: float = 0.001, 
                 method: str = "sdf"):
        """
        Initialize pose filter.
        
        Args:
            input_dir: Directory containing pose data
            collision_threshold: Collision detection threshold
            method: Collision detection method ("sdf" or "mesh")
        """
        self.input_dir = Path(input_dir)
        self.collision_detector = CollisionDetector(collision_threshold)
        self.method = method
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    def get_pose_directories(self) -> List[Path]:
        """Get all pose directories in the input directory."""
        pose_dirs = []
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.startswith("pose_"):
                pose_dirs.append(item)
        
        return sorted(pose_dirs)
    
    def check_pose_collision(self, pose_dir: Path) -> Tuple[bool, float, Dict]:
        """
        Check if a pose has collision.
        
        Args:
            pose_dir: Path to pose directory
            
        Returns:
            Tuple of (collision_detected, min_distance, collision_info)
        """
        moving_file = pose_dir / "moving_object.ply"
        static_file = pose_dir / "static_objects.ply"
        
        if not moving_file.exists() or not static_file.exists():
            return True, 0.0, {"reason": "missing_files", "method": self.method}
        
        try:
            # Load point clouds
            moving_pcd = o3d.io.read_point_cloud(str(moving_file))
            static_pcd = o3d.io.read_point_cloud(str(static_file))
            
            if len(moving_pcd.points) == 0 or len(static_pcd.points) == 0:
                return True, 0.0, {"reason": "empty_pointcloud", "method": self.method}
            
            # Detect collision based on method
            if self.method == "mesh":
                return self.collision_detector.detect_collision_mesh(moving_pcd, static_pcd)
            else:
                return self.collision_detector.detect_collision_sdf(moving_pcd, static_pcd)
                
        except Exception as e:
            print(f"Error processing {pose_dir}: {e}")
            return True, 0.0, {"reason": "processing_error", "error": str(e), "method": self.method}
    
    def filter_poses(self, output_dir: Optional[str] = None, 
                    copy_files: bool = True) -> Tuple[List[Path], List[Path], Dict]:
        """
        Filter poses and optionally copy valid poses to output directory.
        
        Args:
            output_dir: Output directory for filtered poses (optional)
            copy_files: Whether to copy valid pose files to output directory
            
        Returns:
            Tuple of (valid_poses, collision_poses, summary_stats)
        """
        pose_dirs = self.get_pose_directories()
        valid_poses = []
        collision_poses = []
        
        print(f"Processing {len(pose_dirs)} poses...")
        
        # Statistics
        collision_distances = []
        collision_ratios = []
        processing_times = []
        
        for i, pose_dir in enumerate(pose_dirs):
            start_time = datetime.now()
            
            collision_detected, min_distance, collision_info = self.check_pose_collision(pose_dir)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_times.append(processing_time)
            
            if collision_detected:
                collision_poses.append(pose_dir)
            else:
                valid_poses.append(pose_dir)
            
            collision_distances.append(min_distance)
            if "collision_ratio" in collision_info:
                collision_ratios.append(collision_info["collision_ratio"])
            
            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(pose_dirs):
                print(f"  Processed {i + 1}/{len(pose_dirs)} poses "
                      f"(Valid: {len(valid_poses)}, Collisions: {len(collision_poses)})")
        
        # Summary statistics
        summary_stats = {
            "total_poses": len(pose_dirs),
            "valid_poses": len(valid_poses),
            "collision_poses": len(collision_poses),
            "valid_ratio": len(valid_poses) / len(pose_dirs) if pose_dirs else 0,
            "collision_threshold": self.collision_detector.collision_threshold,
            "method": self.method,
            "distance_stats": {
                "min": float(np.min(collision_distances)) if collision_distances else None,
                "max": float(np.max(collision_distances)) if collision_distances else None,
                "mean": float(np.mean(collision_distances)) if collision_distances else None,
                "median": float(np.median(collision_distances)) if collision_distances else None,
            },
            "processing_time": {
                "total": sum(processing_times),
                "average": np.mean(processing_times) if processing_times else 0,
            }
        }
        
        if collision_ratios:
            summary_stats["collision_ratio_stats"] = {
                "min": float(np.min(collision_ratios)),
                "max": float(np.max(collision_ratios)),
                "mean": float(np.mean(collision_ratios)),
                "median": float(np.median(collision_ratios)),
            }
        
        # Copy valid poses to output directory if requested
        if output_dir and copy_files:
            self._copy_valid_poses(valid_poses, output_dir, summary_stats)
        
        return valid_poses, collision_poses, summary_stats
    
    def _copy_valid_poses(self, valid_poses: List[Path], output_dir: str, 
                         summary_stats: Dict):
        """Copy valid poses to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(valid_poses)} valid poses to {output_dir}...")
        
        for i, pose_dir in enumerate(valid_poses):
            # Create new pose directory name
            new_pose_name = f"pose_{i:03d}"
            dest_dir = output_path / new_pose_name
            
            # Copy entire pose directory
            shutil.copytree(pose_dir, dest_dir, dirs_exist_ok=True)
            
            # Update metadata with new pose index
            metadata_file = dest_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata["original_pose_index"] = metadata.get("pose_index", 0)
                metadata["pose_index"] = i
                metadata["filtered"] = True
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Save filtering summary
        summary_file = output_path / "filtering_summary.json"
        summary_stats["input_directory"] = str(self.input_dir)
        summary_stats["output_directory"] = str(output_path)
        summary_stats["filtered_timestamp"] = datetime.now().isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Filtering complete! Valid poses saved to: {output_dir}")
        print(f"Summary saved to: {summary_file}")


def main():
    """Main function for collision filtering."""
    parser = argparse.ArgumentParser(description="Filter poses with collision detection")
    parser.add_argument("input_dir", help="Input directory containing pose data")
    parser.add_argument("-o", "--output", help="Output directory for filtered poses")
    parser.add_argument("-t", "--threshold", type=float, default=0.001,
                       help="Collision detection threshold in meters (default: 0.001)")
    parser.add_argument("-m", "--method", choices=["sdf", "mesh"], default="sdf",
                       help="Collision detection method (default: sdf)")
    parser.add_argument("--no-copy", action="store_true",
                       help="Don't copy files, just analyze collisions")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize pose filter
    pose_filter = PoseFilter(
        input_dir=args.input_dir,
        collision_threshold=args.threshold,
        method=args.method
    )
    
    # Generate output directory name if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        input_name = Path(args.input_dir).name
        args.output = f"outputs/filtered_{input_name}_{timestamp}"
    
    print(f"=== POSE COLLISION FILTERING ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output}")
    print(f"Collision threshold: {args.threshold}m")
    print(f"Detection method: {args.method}")
    print(f"Copy files: {not args.no_copy}")
    print()
    
    # Filter poses
    valid_poses, collision_poses, summary_stats = pose_filter.filter_poses(
        output_dir=args.output if not args.no_copy else None,
        copy_files=not args.no_copy
    )
    
    # Print results
    print(f"\n=== FILTERING RESULTS ===")
    print(f"Total poses processed: {summary_stats['total_poses']}")
    print(f"Valid poses (no collision): {summary_stats['valid_poses']}")
    print(f"Collision poses (filtered out): {summary_stats['collision_poses']}")
    print(f"Valid pose ratio: {summary_stats['valid_ratio']:.2%}")
    
    if summary_stats['distance_stats']['min'] is not None:
        dist_stats = summary_stats['distance_stats']
        print(f"\nDistance statistics:")
        print(f"  Min distance: {dist_stats['min']:.6f}m")
        print(f"  Max distance: {dist_stats['max']:.6f}m")
        print(f"  Mean distance: {dist_stats['mean']:.6f}m")
        print(f"  Median distance: {dist_stats['median']:.6f}m")
    
    if 'collision_ratio_stats' in summary_stats:
        ratio_stats = summary_stats['collision_ratio_stats']
        print(f"\nCollision ratio statistics:")
        print(f"  Min ratio: {ratio_stats['min']:.4f}")
        print(f"  Max ratio: {ratio_stats['max']:.4f}")
        print(f"  Mean ratio: {ratio_stats['mean']:.4f}")
        print(f"  Median ratio: {ratio_stats['median']:.4f}")
    
    proc_stats = summary_stats['processing_time']
    print(f"\nProcessing time:")
    print(f"  Total: {proc_stats['total']:.2f} seconds")
    print(f"  Average per pose: {proc_stats['average']:.4f} seconds")
    
    if args.verbose and collision_poses:
        print(f"\nCollision poses (first 10):")
        for pose in collision_poses[:10]:
            print(f"  {pose.name}")
        if len(collision_poses) > 10:
            print(f"  ... and {len(collision_poses) - 10} more")


if __name__ == "__main__":
    main()