"""
Pose ranking and interactive selection utilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from ..core.utils import calculate_rotation_difference


def rank_poses_by_rotation_and_translation(pose_data_list: List[dict], target_orientation: List[float], target_position: np.ndarray, rotation_weight: float = 0.5, translation_weight: float = 0.5) -> List[Tuple[dict, float, float, float]]:
    """Rank poses by both rotation and translation similarity to target pose.
    
    Args:
        pose_data_list: List of pose data dictionaries
        target_orientation: Target orientation [w, x, y, z] for comparison
        target_position: Target position [x, y, z] for comparison
        rotation_weight: Weight for rotation component (0-1)
        translation_weight: Weight for translation component (0-1)
        
    Returns:
        List of (pose_data, combined_score, rotation_difference, translation_distance) tuples sorted by combined score
    """
    print(f"Ranking {len(pose_data_list)} poses by rotation and translation similarity...")
    
    pose_rankings = []
    for pose_data in pose_data_list:
        # Calculate rotation difference
        rotation_diff = calculate_rotation_difference(target_orientation, pose_data['orientation'])
        
        # Calculate translation distance
        translation_distance = np.linalg.norm(np.array(pose_data['position']) - target_position)
        
        # Normalize scores (rotation is in degrees, translation in meters)
        # Normalize rotation to 0-1 (assuming max meaningful difference is 180°)
        normalized_rotation = rotation_diff / 180.0
        
        # Normalize translation to 0-1 (assuming max meaningful distance is 0.5m)
        normalized_translation = min(translation_distance / 0.5, 1.0)
        
        # Calculate combined score (lower is better)
        combined_score = rotation_weight * normalized_rotation + translation_weight * normalized_translation
        
        pose_rankings.append((pose_data, combined_score, rotation_diff, translation_distance))
    
    # Sort by combined score (ascending - smaller scores are better)
    pose_rankings.sort(key=lambda x: x[1])
    
    print(f"Pose ranking complete. Top 10 poses with best combined scores:")
    for i, (pose_data, combined_score, rot_diff, trans_dist) in enumerate(pose_rankings[:10]):
        print(f"  {i+1:2d}. Pose {pose_data['index']:03d}: Score={combined_score:.3f} (Rot={rot_diff:.1f}°, Trans={trans_dist*1000:.1f}mm)")
    
    if len(pose_rankings) > 10:
        print(f"  ... and {len(pose_rankings) - 10} more poses")
    
    return pose_rankings


def rank_poses_by_rotation(pose_data_list: List[dict], target_orientation: List[float]) -> List[Tuple[dict, float]]:
    """Rank poses by rotation similarity to target orientation (legacy function for backward compatibility).
    
    Args:
        pose_data_list: List of pose data dictionaries
        target_orientation: Target orientation [w, x, y, z] for comparison
        
    Returns:
        List of (pose_data, rotation_difference) tuples sorted by rotation difference
    """
    print(f"Ranking {len(pose_data_list)} poses by rotation similarity...")
    
    pose_rankings = []
    for pose_data in pose_data_list:
        rotation_diff = calculate_rotation_difference(target_orientation, pose_data['orientation'])
        pose_rankings.append((pose_data, rotation_diff))
    
    # Sort by rotation difference (ascending - smaller differences first)
    pose_rankings.sort(key=lambda x: x[1])
    
    print(f"Pose ranking complete. Top 10 poses with smallest rotation differences:")
    for i, (pose_data, rot_diff) in enumerate(pose_rankings[:10]):
        print(f"  {i+1:2d}. Pose {pose_data['index']:03d}: {rot_diff:.2f}°")
    
    if len(pose_rankings) > 10:
        print(f"  ... and {len(pose_rankings) - 10} more poses")
    
    return pose_rankings


def interactive_pose_selection(pose_rankings: List[Tuple[Any, ...]], max_display: int = 10) -> Optional[int]:
    """Interactive pose selection from ranked poses.
    
    Args:
        pose_rankings: List of pose ranking tuples (can be rotation-only or combined scoring)
        max_display: Maximum number of poses to display
        
    Returns:
        Selected pose index (1-based) or None if 0 is selected
    """
    print(f"\n=== INTERACTIVE POSE SELECTION ===")
    
    # Check if this is combined scoring (4 elements) or rotation-only (2 elements)
    is_combined_scoring = len(pose_rankings[0]) == 4
    
    if is_combined_scoring:
        print(f"Available poses ranked by combined rotation and translation similarity:")
        print(f"(Lower scores = better matches)\n")
    else:
        print(f"Available poses ranked by rotation similarity to final waypoint:")
        print(f"(Smaller rotation differences = better matches)\n")
    
    # Display top poses
    display_count = min(max_display, len(pose_rankings))
    for i in range(display_count):
        if is_combined_scoring:
            pose_data, combined_score, rot_diff, trans_dist = pose_rankings[i]
            print(f"  {i+1:2d}. Pose {pose_data['index']:03d}: Score={combined_score:.3f} (Rot={rot_diff:.1f}°, Trans={trans_dist*1000:.1f}mm)")
        else:
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
                    if is_combined_scoring:
                        print(f"Selected pose {choice_int}: Score={selected_pose[1]:.3f} (Rot={selected_pose[2]:.1f}°, Trans={selected_pose[3]*1000:.1f}mm)")
                    else:
                        print(f"Selected pose {choice_int}: {selected_pose[1]:.2f}° rotation difference")
                    return choice_int
            else:
                print(f"Please enter a number between 0 and {len(pose_rankings)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None