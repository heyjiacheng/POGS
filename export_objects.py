#!/usr/bin/env python3
"""
Safe script to export individual objects from POGS gaussian splats
Avoids the CUDA errors in the viewer by working directly with the data
"""

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
import json
import argparse

def load_clusters(clusters_file):
    """Load cluster information from the clusters.npy file"""
    clusters = np.load(clusters_file, allow_pickle=True)
    cluster_labels = clusters[0]  # Cluster labels for each gaussian
    keep_indices = clusters[1]    # Indices of gaussians to keep
    
    return cluster_labels, keep_indices

def load_gaussian_data(ply_file):
    """Load gaussian data from PLY file"""
    import struct
    
    with open(ply_file, 'rb') as f:
        # Read header
        header = b''
        while True:
            line = f.readline()
            header += line
            if line.strip() == b'end_header':
                break
        
        # Parse header to get number of vertices
        header_str = header.decode('utf-8')
        for line in header_str.split('\n'):
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                break
        
        # Each vertex has 62 floats: x,y,z,nx,ny,nz,f_dc_0-2,f_rest_0-44,opacity,scale_0-2,rot_0-3
        vertex_size = 4 * 62  # 248 bytes per vertex
        
        gaussians = []
        for i in range(num_vertices):
            vertex_data = f.read(vertex_size)
            if len(vertex_data) != vertex_size:
                break
            
            values = struct.unpack('<' + 'f' * 62, vertex_data)
            
            # Extract relevant data
            gaussian = {
                'position': np.array([values[0], values[1], values[2]]),
                'normal': np.array([values[3], values[4], values[5]]),
                'color_dc': np.array([values[6], values[7], values[8]]),
                'color_rest': np.array(values[9:54]),  # f_rest_0 to f_rest_44
                'opacity': values[54],
                'scale': np.array([values[55], values[56], values[57]]),
                'rotation': np.array([values[58], values[59], values[60], values[61]])
            }
            gaussians.append(gaussian)
    
    return gaussians

def export_object_ply(gaussians, object_indices, output_file):
    """Export specific object gaussians to PLY file"""
    
    # Create point cloud for the object
    positions = np.array([gaussians[i]['position'] for i in object_indices])
    
    # Create colors from DC coefficients (simplified)
    colors = np.array([gaussians[i]['color_dc'] for i in object_indices])
    # Convert from spherical harmonics DC to RGB (simplified)
    colors = (colors + 0.5) * 255  # Simple conversion
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Save as PLY
    o3d.io.write_point_cloud(str(output_file), pcd)
    
    print(f"Exported {len(object_indices)} gaussians to {output_file}")
    
    # Also save detailed gaussian data as numpy
    object_data = {
        'positions': positions,
        'colors_dc': np.array([gaussians[i]['color_dc'] for i in object_indices]),
        'colors_rest': np.array([gaussians[i]['color_rest'] for i in object_indices]),
        'opacities': np.array([gaussians[i]['opacity'] for i in object_indices]),
        'scales': np.array([gaussians[i]['scale'] for i in object_indices]),
        'rotations': np.array([gaussians[i]['rotation'] for i in object_indices]),
        'normals': np.array([gaussians[i]['normal'] for i in object_indices])
    }
    
    np.save(str(output_file).replace('.ply', '_detailed.npy'), object_data)
    print(f"Saved detailed data to {str(output_file).replace('.ply', '_detailed.npy')}")

def main():
    parser = argparse.ArgumentParser(description='Export individual objects from POGS gaussian splats')
    parser.add_argument('--clusters', type=str, default='outputs/box/clusters.npy', 
                       help='Path to clusters.npy file')
    parser.add_argument('--gaussians', type=str, default='outputs/box/prime_full_gaussians.ply',
                       help='Path to full gaussians PLY file')
    parser.add_argument('--output-dir', type=str, default='exports/objects',
                       help='Output directory for object files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cluster information
    try:
        cluster_labels, keep_indices = load_clusters(args.clusters)
        print(f"Loaded {len(cluster_labels)} cluster labels")
        print(f"Unique clusters: {np.unique(cluster_labels)}")
    except Exception as e:
        print(f"Error loading clusters: {e}")
        return
    
    # Load gaussian data
    try:
        gaussians = load_gaussian_data(args.gaussians)
        print(f"Loaded {len(gaussians)} gaussians")
    except Exception as e:
        print(f"Error loading gaussians: {e}")
        return
    
    # Export each object
    unique_clusters = np.unique(cluster_labels)
    print(f"Number of loaded gaussians: {len(gaussians)}")
    print(f"Number of cluster labels: {len(cluster_labels)}")
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points
            continue
            
        # Find indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        object_indices = np.where(cluster_mask)[0]
        
        # Filter indices to only include valid ones
        valid_indices = object_indices[object_indices < len(gaussians)]
        
        if len(valid_indices) == 0:
            print(f"No valid indices for cluster {cluster_id}")
            continue
            
        print(f"Cluster {cluster_id}: {len(valid_indices)} valid gaussians out of {len(object_indices)} total")
        
        # Export this object
        output_file = output_dir / f"object_{cluster_id}.ply"
        export_object_ply(gaussians, valid_indices, output_file)
        
        # Save object info
        info = {
            'cluster_id': int(cluster_id),
            'num_gaussians': len(valid_indices),
            'center': np.mean([gaussians[i]['position'] for i in valid_indices], axis=0).tolist(),
            'bbox_min': np.min([gaussians[i]['position'] for i in valid_indices], axis=0).tolist(),
            'bbox_max': np.max([gaussians[i]['position'] for i in valid_indices], axis=0).tolist()
        }
        
        with open(output_dir / f"object_{cluster_id}_info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    print(f"Exported {len(unique_clusters)} objects to {output_dir}")

if __name__ == "__main__":
    main()