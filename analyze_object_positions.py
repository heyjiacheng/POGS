#!/usr/bin/env python3
"""
Analyze positions and properties of individual objects from exported gaussian splats
"""

import numpy as np
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_object(object_file, info_file):
    """Analyze a single object's gaussian positions and properties"""
    
    # Load detailed data
    data = np.load(object_file, allow_pickle=True).item()
    
    # Load info
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    print(f"\n=== Object {info['cluster_id']} Analysis ===")
    print(f"Number of gaussians: {info['num_gaussians']}")
    print(f"Center: {info['center']}")
    print(f"Bounding box min: {info['bbox_min']}")
    print(f"Bounding box max: {info['bbox_max']}")
    
    # Analyze positions
    positions = data['positions']
    print(f"\nPosition Statistics:")
    print(f"  Mean: {np.mean(positions, axis=0)}")
    print(f"  Std:  {np.std(positions, axis=0)}")
    print(f"  Min:  {np.min(positions, axis=0)}")
    print(f"  Max:  {np.max(positions, axis=0)}")
    
    # Analyze other properties
    opacities = data['opacities']
    scales = data['scales']
    
    print(f"\nOpacity Statistics:")
    print(f"  Mean: {np.mean(opacities):.6f}")
    print(f"  Std:  {np.std(opacities):.6f}")
    print(f"  Min:  {np.min(opacities):.6f}")
    print(f"  Max:  {np.max(opacities):.6f}")
    
    print(f"\nScale Statistics:")
    print(f"  Mean: {np.mean(scales, axis=0)}")
    print(f"  Std:  {np.std(scales, axis=0)}")
    print(f"  Min:  {np.min(scales, axis=0)}")
    print(f"  Max:  {np.max(scales, axis=0)}")
    
    return positions, data

def visualize_object_positions(positions, object_id, save_plot=False):
    """Create 3D visualization of object positions"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization (max 10000 points)
    if len(positions) > 10000:
        indices = np.random.choice(len(positions), 10000, replace=False)
        vis_positions = positions[indices]
    else:
        vis_positions = positions
    
    # Create scatter plot
    ax.scatter(vis_positions[:, 0], vis_positions[:, 1], vis_positions[:, 2], 
               s=1, alpha=0.6, c='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Object {object_id} - Gaussian Positions')
    
    if save_plot:
        plt.savefig(f'object_{object_id}_positions.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot to object_{object_id}_positions.png")
    
    plt.show()

def compare_objects(objects_dir):
    """Compare all objects side by side"""
    objects_dir = Path(objects_dir)
    
    # Find all object files
    detail_files = list(objects_dir.glob("object_*_detailed.npy"))
    
    print(f"\n=== Comparing {len(detail_files)} Objects ===")
    
    all_positions = []
    all_labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, detail_file in enumerate(detail_files):
        # Extract object ID from filename
        object_id = detail_file.stem.split('_')[1]
        
        # Load data
        data = np.load(detail_file, allow_pickle=True).item()
        positions = data['positions']
        
        # Sample for visualization
        if len(positions) > 2000:
            indices = np.random.choice(len(positions), 2000, replace=False)
            sampled_positions = positions[indices]
        else:
            sampled_positions = positions
        
        all_positions.append(sampled_positions)
        all_labels.extend([object_id] * len(sampled_positions))
        
        print(f"Object {object_id}: {len(positions)} gaussians")
    
    # Create combined visualization
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (positions, color) in enumerate(zip(all_positions, colors[:len(all_positions)])):
        object_id = detail_files[i].stem.split('_')[1]
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  s=1, alpha=0.6, c=color, label=f'Object {object_id}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('All Objects - Gaussian Positions')
    ax.legend()
    
    plt.savefig('all_objects_positions.png', dpi=150, bbox_inches='tight')
    print("Saved combined plot to all_objects_positions.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze gaussian positions in exported objects')
    parser.add_argument('--objects-dir', type=str, default='exports/objects',
                       help='Directory containing exported objects')
    parser.add_argument('--object-id', type=str, default=None,
                       help='Specific object ID to analyze (e.g., "0.0")')
    parser.add_argument('--visualize', action='store_true',
                       help='Create 3D visualizations')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all objects')
    
    args = parser.parse_args()
    
    objects_dir = Path(args.objects_dir)
    
    if args.object_id:
        # Analyze specific object
        detail_file = objects_dir / f"object_{args.object_id}_detailed.npy"
        info_file = objects_dir / f"object_{args.object_id}_info.json"
        
        if detail_file.exists() and info_file.exists():
            positions, data = analyze_object(detail_file, info_file)
            
            if args.visualize:
                visualize_object_positions(positions, args.object_id, save_plot=True)
        else:
            print(f"Object {args.object_id} files not found")
    else:
        # Analyze all objects
        detail_files = list(objects_dir.glob("object_*_detailed.npy"))
        
        for detail_file in detail_files:
            object_id = detail_file.stem.split('_')[1]
            info_file = objects_dir / f"object_{object_id}_info.json"
            
            if info_file.exists():
                positions, data = analyze_object(detail_file, info_file)
    
    if args.compare:
        compare_objects(objects_dir)

if __name__ == "__main__":
    main()