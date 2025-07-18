import struct
import numpy as np

# Read the PLY file to understand the format
with open('/home/jiachengxu/workspace/master_thesis/POGS/exports/splat/splat.ply', 'rb') as f:
    # Read header
    header = b''
    while True:
        line = f.readline()
        header += line
        if line.strip() == b'end_header':
            break
    
    # Count properties: x,y,z,nx,ny,nz,f_dc_0-2,f_rest_0-44,opacity,scale_0-2,rot_0-3
    # That's 3+3+3+45+1+3+4 = 62 floats per vertex
    vertex_size = 4 * 62
    print(f'Corrected vertex size: {vertex_size} bytes')
    
    # Read first vertex
    vertex_data = f.read(vertex_size)
    if len(vertex_data) != vertex_size:
        print(f'Warning: Read {len(vertex_data)} bytes instead of {vertex_size}')
    
    values = struct.unpack('<' + 'f' * (len(vertex_data) // 4), vertex_data)
    
    print(f'First vertex data:')
    print(f'Position (x,y,z): {values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f}')
    print(f'Normal (nx,ny,nz): {values[3]:.6f}, {values[4]:.6f}, {values[5]:.6f}')
    print(f'Color DC (f_dc_0-2): {values[6]:.6f}, {values[7]:.6f}, {values[8]:.6f}')
    print(f'Opacity: {values[51]:.6f}')
    print(f'Scale: {values[52]:.6f}, {values[53]:.6f}, {values[54]:.6f}')
    print(f'Rotation: {values[55]:.6f}, {values[56]:.6f}, {values[57]:.6f}, {values[58]:.6f}')
    
    # Read all positions
    f.seek(0)
    # Skip header
    while True:
        line = f.readline()
        if line.strip() == b'end_header':
            break
    
    print(f'\nReading all {1055483} gaussian positions...')
    positions = []
    for i in range(min(10, 1055483)):  # Show first 10 positions
        vertex_data = f.read(vertex_size)
        if len(vertex_data) != vertex_size:
            break
        values = struct.unpack('<' + 'f' * (len(vertex_data) // 4), vertex_data)
        positions.append([values[0], values[1], values[2]])
    
    print(f'First 10 gaussian positions:')
    for i, pos in enumerate(positions):
        print(f'Gaussian {i+1}: x={pos[0]:.6f}, y={pos[1]:.6f}, z={pos[2]:.6f}')