#!/usr/bin/env python3
"""
Show detailed positions of individual gaussians in each object
"""

import numpy as np
import json
from pathlib import Path

def show_gaussian_positions(objects_dir, num_gaussians=10):
    """Show positions of first few gaussians in each object"""
    
    objects_dir = Path(objects_dir)
    detail_files = sorted(list(objects_dir.glob("object_*_detailed.npy")))
    
    print(f"=== Showing first {num_gaussians} gaussian positions for each object ===\n")
    
    for detail_file in detail_files:
        # Extract object ID from filename
        object_id = detail_file.stem.split('_')[1]
        
        # Load data
        data = np.load(detail_file, allow_pickle=True).item()
        positions = data['positions']
        opacities = data['opacities']
        scales = data['scales']
        rotations = data['rotations']
        
        # Load info
        info_file = objects_dir / f"object_{object_id}_info.json"
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print(f"🎯 Object {object_id} ({info['num_gaussians']} total gaussians)")
        print(f"   Center: [{info['center'][0]:.4f}, {info['center'][1]:.4f}, {info['center'][2]:.4f}]")
        print(f"   Bounding box: [{info['bbox_min'][0]:.2f}, {info['bbox_min'][1]:.2f}, {info['bbox_min'][2]:.2f}] to [{info['bbox_max'][0]:.2f}, {info['bbox_max'][1]:.2f}, {info['bbox_max'][2]:.2f}]")
        print()
        
        # Show individual gaussian details
        for i in range(min(num_gaussians, len(positions))):
            pos = positions[i]
            opacity = opacities[i]
            scale = scales[i]
            rotation = rotations[i]
            
            print(f"   Gaussian {i+1:2d}:")
            print(f"     Position: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
            print(f"     Opacity:  {opacity:8.4f}")
            print(f"     Scale:    [{scale[0]:8.4f}, {scale[1]:8.4f}, {scale[2]:8.4f}]")
            print(f"     Rotation: [{rotation[0]:6.3f}, {rotation[1]:6.3f}, {rotation[2]:6.3f}, {rotation[3]:6.3f}]")
            print()
        
        print("-" * 80)
        print()

if __name__ == "__main__":
    show_gaussian_positions("exports/objects", num_gaussians=5)