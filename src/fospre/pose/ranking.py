"""
Pose ranking and interactive selection utilities.
"""

from typing import List, Tuple, Optional, Any
from ..core.utils import calculate_rotation_difference


def rank_poses_by_rotation(pose_data_list: List[dict], target_orientation: List[float]) -> List[Tuple[dict, float]]:
    """Rank poses by rotation similarity to target orientation.
    
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