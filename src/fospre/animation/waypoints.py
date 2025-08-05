"""
Waypoint management and conversion utilities.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from ..pose.transforms import get_ee_target_relative_transform, apply_relative_transform


def load_waypoints_from_json(waypoints_file: str) -> List[Tuple[float, float, float]]:
    """Load waypoints from JSON file.
    
    Note: These are end effector waypoints, not target object waypoints.
    For target object animation, we need to use the target object's own initial pose.
    
    Args:
        waypoints_file: Path to waypoints JSON file
        
    Returns:
        List of waypoint positions as (x, y, z) tuples
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
        from scipy.spatial.transform import Rotation as R
        
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
            "modification_timestamp": "modified",
            "transformation_note": "End effector pose calculated to maintain relative transform to target object"
        }
        
        print(f"\n=== NEW SUBGOALS CREATED ===")
        print(f"Original target object pose: [{original_target_pose[0]:.6f}, {original_target_pose[1]:.6f}, {original_target_pose[2]:.6f}]")
        print(f"New target object pose:      [{new_target_pose[0]:.6f}, {new_target_pose[1]:.6f}, {new_target_pose[2]:.6f}]")
        print(f"Selected pose index:         {selected_pose_data['index']}")
        if 'rotation_difference_deg' in selected_pose_data:
            print(f"Rotation difference:         {selected_pose_data['rotation_difference_deg']:.2f}°")
    else:
        # Keep original waypoints
        new_data['metadata'] = {
            "modified": False,
            "original_file": original_waypoints_file,
            "modification_timestamp": "modified"
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