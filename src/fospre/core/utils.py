"""
General utility functions for the POGS animation pipeline.
"""

import time
import numpy as np
from pathlib import Path
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def find_latest_config(base_path: str = "outputs/box/pogs") -> Path:
    """Find the latest config.yml file in the pogs output directories.
    
    Args:
        base_path: Base directory to search for config files
        
    Returns:
        Path to the latest config.yml file
        
    Raises:
        FileNotFoundError: If no config files are found
    """
    base_dir = Path(base_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_path} does not exist")
    
    # Find all config.yml files in timestamped directories
    config_files = list(base_dir.glob("*/config.yml"))
    
    if not config_files:
        raise FileNotFoundError(f"No config.yml files found in {base_path}")
    
    # Sort by directory name (which contains timestamp) and get the latest
    config_files.sort(key=lambda x: x.parent.name, reverse=True)
    return config_files[0]


def calculate_rotation_difference(quat1: np.ndarray, quat2: np.ndarray) -> float:
    """Calculate the rotation difference between two quaternions in degrees.
    
    Args:
        quat1: First quaternion [w, x, y, z]
        quat2: Second quaternion [w, x, y, z]
        
    Returns:
        Rotation difference in degrees
    """
    # Convert to scipy Rotation objects
    rot1_xyzw = np.array([quat1[1], quat1[2], quat1[3], quat1[0]])
    rot2_xyzw = np.array([quat2[1], quat2[2], quat2[3], quat2[0]])
    
    rot1 = R.from_quat(rot1_xyzw)
    rot2 = R.from_quat(rot2_xyzw)
    
    # Calculate relative rotation
    relative_rotation = rot2 * rot1.inv()
    
    # Get rotation angle in radians and convert to degrees
    angle_rad = relative_rotation.magnitude()
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def safe_sleep(seconds: float = 2.0):
    """Safe sleep function for pipeline operations."""
    time.sleep(seconds)