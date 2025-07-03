import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import yaml
import argparse
import pathlib
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform, DepthImage, CameraIntrinsics, PointCloud
import open3d as o3d
import json
from tqdm import tqdm
import pyrealsense2 as rs

# ===============================
# Setup paths and transformation
# ===============================

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = os.path.join(DIR_PATH, '../data/utils/datasets')
calibration_save_path = os.path.join(DIR_PATH, '../calibration_outputs')
config_filepath = os.path.join(DIR_PATH, '../configs/camera_config.yaml')

# Transformation matrix to convert between NeRF coordinate frame and image coordinate frame
nerf_frame_to_image_frame = np.array([[1,0,0,0],
                                        [0,-1,0,0],
                                        [0,0,-1,0],
                                        [0,0,0,1]])

# Load calibrated transforms
try:
    wrist_to_d435 = RigidTransform.load(os.path.join(calibration_save_path, 'wrist_to_d435.tf'))
    world_to_d405 = RigidTransform.load(os.path.join(calibration_save_path, 'world_to_d405.tf'))
    print("✓ Loaded calibration files successfully")
except FileNotFoundError as e:
    print(f"✗ Error loading calibration files: {e}")
    print("Please run calibrate_cameras.py first to generate calibration files")
    exit(1)

# ===============================
# RealSense Camera Wrapper
# ===============================

class RealSenseCamera:
    def __init__(self, serial_number, width=640, height=480, fps=30, camera_name="realsense"):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start pipeline and get camera info
        profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        self.intrinsics = CameraIntrinsics(camera_name, self.intr.width, self.intr.height,
                                           self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
        
        # Warm up camera
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def get_rgb_depth(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        
        color_image = np.asanyarray(color.get_data())
        depth_image = np.asanyarray(depth.get_data()) / 1000.0  # Convert to meters
        
        # Convert BGR to RGB for consistency
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return color_image, depth_image

    def get_ns_intrinsics(self):
        """Return intrinsics in NeRFStudio format"""
        return {
            "fl_x": self.intr.fx,
            "fl_y": self.intr.fy,
            "cx": self.intr.ppx,
            "cy": self.intr.ppy,
            "w": self.intr.width,
            "h": self.intr.height,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0
        }

    def stop(self):
        self.pipeline.stop()

# ===============================
# Point Cloud Generation
# ===============================

def generate_pointcloud(depth, intrinsics):
    """Generate point cloud from depth image using camera intrinsics"""
    depth_im = DepthImage(depth, frame=intrinsics.frame)
    pc = intrinsics.deproject(depth_im)
    return PointCloud(pc.data, frame=intrinsics.frame)

# ===============================
# Helper Functions
# ===============================

def setup_directories(scene_name):
    """Create directory structure for saving data"""
    dirs = {
        "rgb": os.path.join(HOME_DIR, scene_name, "rgb"),
        "depth": os.path.join(HOME_DIR, scene_name, "depth"),
        "poses": os.path.join(HOME_DIR, scene_name, "poses"),
        "scene": os.path.join(HOME_DIR, scene_name)
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_pose_nerf(transform, frame_idx, save_dir):
    """Save camera pose in NeRF format"""
    # Convert to NeRF coordinate system (flip y and z axes)
    nerf_transform = transform.matrix @ nerf_frame_to_image_frame
    nerf_rigid = RigidTransform(
        rotation=nerf_transform[:3, :3],
        translation=nerf_transform[:3, 3],
        from_frame="nerf_cam",
        to_frame="world"
    )
    
    np.savetxt(os.path.join(save_dir, f"{frame_idx:03d}.txt"), nerf_rigid.matrix)
    return nerf_rigid.matrix

def save_transforms_json(save_dirs, intrinsics_list, scene_name):
    """Save transforms.json in NeRF format"""
    data = {
        "frames": [],
        "ply_file_path": "sparse_pc.ply"
    }
    
    pose_files = sorted([f for f in os.listdir(save_dirs["poses"]) if f.endswith('.txt')])
    
    for i, (pose_file, intrinsics) in enumerate(zip(pose_files, intrinsics_list)):
        transform_matrix = np.loadtxt(os.path.join(save_dirs["poses"], pose_file))
        
        frame_data = intrinsics.copy()
        frame_data.update({
            "file_path": f"rgb/frame_{i+1:05d}.png",
            "depth_file_path": f"depth/frame_{i+1:05d}.npy",
            "transform_matrix": transform_matrix.tolist()
        })
        data["frames"].append(frame_data)
    
    with open(os.path.join(save_dirs["scene"], "transforms.json"), 'w') as f:
        json.dump(data, f, indent=4)

def clear_tcp(robot):
    """Clear robot TCP settings"""
    tcp = RigidTransform(translation=np.array([0, 0, 0]), from_frame='tool', to_frame='wrist')
    robot.set_tcp(tcp)

# ===============================
# Main Scene Capture Function
# ===============================

def main(scene_name="realsense_scene"):
    """
    Main function for capturing multi-view scene data using RealSense cameras
    
    Args:
        scene_name: Name for the captured scene dataset
    """
    print(f"Starting scene capture for: {scene_name}")
    
    # Load camera configuration
    try:
        with open(config_filepath, 'r') as f:
            camera_config = yaml.safe_load(f)
        print("✓ Loaded camera configuration")
    except FileNotFoundError:
        print(f"✗ Camera config file not found: {config_filepath}")
        print("Please ensure camera_config.yaml exists with camera serial numbers")
        return
    
    # Setup directories
    save_dirs = setup_directories(scene_name)
    
    # Initialize robot
    print("Initializing robot...")
    robot = UR5Robot(gripper=1)
    clear_tcp(robot)
    
    # Move to home position
    home_joints = np.array([0.11, -2.23, 1.38, -0.71, -1.56, 1.68])
    robot.move_joint(home_joints, vel=1.0, acc=0.1)
    robot.gripper.open()
    print("✓ Robot initialized and moved to home position")
    
    # Initialize cameras
    print("Initializing cameras...")
    try:
        wrist_cam = RealSenseCamera(
            serial_number=camera_config['wrist_d435']['id'],
            width=640, height=480, fps=30, camera_name="d435"
        )
        third_cam = RealSenseCamera(
            serial_number=camera_config['static_d405']['id'],
            width=640, height=480, fps=30, camera_name="d405"
        )
        print("✓ Cameras initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing cameras: {e}")
        return
    
    # Storage for data
    global_pc = []
    global_rgb = []
    intrinsics_list = []
    
    # Capture from static third-person D405 first
    print("Capturing from static D405 camera...")
    color_d405, depth_d405 = third_cam.get_rgb_depth()
    pc_d405 = generate_pointcloud(depth_d405, third_cam.intrinsics)
    pc_d405_world = world_to_d405 * pc_d405
    
    # Save static camera data
    valid_points = pc_d405_world.data.T
    # Only keep valid points (no NaN values)
    valid_mask = ~np.isnan(valid_points).any(axis=1)
    valid_points = valid_points[valid_mask]
    global_pc.append(valid_points)
    
    # For colors: the point cloud from deproject should have same number of points as valid pixels
    # We need to get the colors that correspond to the point cloud
    height, width = depth_d405.shape
    colors_flat = color_d405.reshape(-1, 3)
    
    # Get colors corresponding to the point cloud points (before NaN filtering)
    original_pc_points = pc_d405_world.data.T  # All points from deproject
    colors_for_pc = colors_flat[:len(original_pc_points)]  # Take same number of colors as points
    
    # Apply the same valid_mask to colors
    if len(valid_points) > 0 and len(colors_for_pc) >= len(valid_mask):
        colors_final = colors_for_pc[valid_mask]
        global_rgb.append(colors_final)
    else:
        # If dimensions don't match, use a simpler approach
        print(f"Warning: Color-point dimension mismatch. Points: {len(valid_points)}, Colors: {len(colors_for_pc)}")
        # Just take the first N colors where N is the number of valid points
        if len(valid_points) > 0:
            colors_simple = colors_flat[:len(valid_points)]
            global_rgb.append(colors_simple)
        else:
            global_rgb.append(np.empty((0, 3)))
    
    # Save first frame
    cv2.imwrite(os.path.join(save_dirs["rgb"], "frame_00001.png"), 
                cv2.cvtColor(color_d405, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(save_dirs["depth"], "frame_00001.npy"), depth_d405)
    
    # Save static camera pose
    static_pose = save_pose_nerf(world_to_d405, 0, save_dirs["poses"])
    intrinsics_list.append(third_cam.get_ns_intrinsics())
    
    # Load trajectory
    trajectory_path = os.path.join(calibration_save_path, "realsense_calibration_trajectory.npy")
    if not os.path.exists(trajectory_path):
        print(f"✗ Trajectory file not found: {trajectory_path}")
        print("Please run calibrate_cameras.py first to generate trajectory")
        wrist_cam.stop()
        third_cam.stop()
        return
    
    trajectory = np.load(trajectory_path)
    print(f"✓ Loaded trajectory with {len(trajectory)} poses")
    
    # Execute trajectory and capture from wrist-mounted D435
    print("Executing trajectory and capturing from wrist D435...")
    for i, joint in enumerate(tqdm(trajectory, desc="Capturing frames")):
        # Move robot to trajectory point
        robot.move_joint(joint, vel=0.7, acc=0.15)
        time.sleep(1.0)  # Wait for stabilization
        
        # Get current robot pose
        wrist_pose = robot.get_pose()
        wrist_pose.from_frame = "wrist"
        wrist_pose.to_frame = "world"
        
        # Calculate camera pose
        cam_pose = wrist_pose * wrist_to_d435
        
        # Capture from wrist camera
        color_d435, depth_d435 = wrist_cam.get_rgb_depth()
        pc_d435 = generate_pointcloud(depth_d435, wrist_cam.intrinsics)
        pc_world = cam_pose * pc_d435
        
        # Store point cloud data
        valid_points = pc_world.data.T
        # Only keep valid points (no NaN values)
        valid_mask = ~np.isnan(valid_points).any(axis=1)
        valid_points = valid_points[valid_mask]
        global_pc.append(valid_points)
        
        # For colors: get colors corresponding to the point cloud
        height, width = depth_d435.shape
        colors_flat = color_d435.reshape(-1, 3)
        
        # Get colors corresponding to the point cloud points (before NaN filtering)
        original_pc_points = pc_world.data.T  # All points from deproject
        colors_for_pc = colors_flat[:len(original_pc_points)]  # Take same number of colors as points
        
        # Apply the same valid_mask to colors
        if len(valid_points) > 0 and len(colors_for_pc) >= len(valid_mask):
            colors_final = colors_for_pc[valid_mask]
            global_rgb.append(colors_final)
        else:
            # If dimensions don't match, use a simpler approach
            print(f"Warning: Color-point dimension mismatch. Points: {len(valid_points)}, Colors: {len(colors_for_pc)}")
            # Just take the first N colors where N is the number of valid points
            if len(valid_points) > 0:
                colors_simple = colors_flat[:len(valid_points)]
                global_rgb.append(colors_simple)
            else:
                global_rgb.append(np.empty((0, 3)))
        
        # Save frame data
        frame_idx = i + 2  # +2 because static camera is frame 1
        cv2.imwrite(os.path.join(save_dirs["rgb"], f"frame_{frame_idx:05d}.png"), 
                    cv2.cvtColor(color_d435, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(save_dirs["depth"], f"frame_{frame_idx:05d}.npy"), depth_d435)
        
        # Save camera pose
        wrist_pose_nerf = save_pose_nerf(cam_pose, i + 1, save_dirs["poses"])
        intrinsics_list.append(wrist_cam.get_ns_intrinsics())
    
    # Generate and save combined point cloud
    print("Generating combined point cloud...")
    
    # Filter out empty arrays before stacking
    valid_pc_arrays = [pc for pc in global_pc if len(pc) > 0]
    valid_rgb_arrays = [rgb for rgb in global_rgb if len(rgb) > 0]
    
    if len(valid_pc_arrays) > 0:
        pc_all = np.vstack(valid_pc_arrays)
        rgb_all = np.vstack(valid_rgb_arrays)
        
        # Subsample for efficiency
        if len(pc_all) > 100000:
            indices = np.random.choice(len(pc_all), 100000, replace=False)
            pc_all = pc_all[indices]
            rgb_all = rgb_all[indices]
    else:
        print("Warning: No valid point cloud data captured")
        pc_all = np.empty((0, 3))
        rgb_all = np.empty((0, 3))
    
    # Save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_all)
    pcd.colors = o3d.utility.Vector3dVector(rgb_all / 255.0)
    o3d.io.write_point_cloud(os.path.join(save_dirs["scene"], "sparse_pc.ply"), pcd)
    
    # Save camera intrinsics
    wrist_cam.intrinsics.save(os.path.join(save_dirs["scene"], "camera_intrinsics.intr"))
    
    # Save transforms.json for NeRF
    save_transforms_json(save_dirs, intrinsics_list, scene_name)
    
    # Cleanup
    wrist_cam.stop()
    third_cam.stop()
    robot.ur_c.disconnect()
    
    print(f"✓ Scene capture completed successfully!")
    print(f"✓ Data saved to: {save_dirs['scene']}")
    print(f"✓ Captured {len(trajectory) + 1} frames total")
    print(f"✓ Point cloud contains {len(pc_all)} points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture multi-view scene data using RealSense cameras")
    parser.add_argument("--scene", type=str, required=True, 
                        help="Name of the scene to capture")
    args = parser.parse_args()
    
    main(args.scene)