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
from sklearn.cluster import DBSCAN

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
    def __init__(self, serial_number, width=640, height=480, fps=15, camera_name="realsense"):  # Reduced FPS from 30 to 15
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_name = camera_name
        self.pipeline = None
        self.config = None
        self.align = None
        self.intr = None
        self.intrinsics = None
        
        # Initialize camera connection
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize camera connection"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_device(self.serial_number)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            # Start pipeline and get camera info
            profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            
            # Get intrinsics
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intr = color_stream.get_intrinsics()
            self.intrinsics = CameraIntrinsics(self.camera_name, self.intr.width, self.intr.height,
                                               self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
            
            # Warm up camera
            for _ in range(10):
                self.pipeline.wait_for_frames()
                
            print(f"✓ Camera {self.serial_number} initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing camera {self.serial_number}: {e}")
            raise

    def _reconnect_camera(self, max_retries=3):
        """Attempt to reconnect the camera"""
        print(f"⚠ Camera {self.serial_number} disconnected, attempting to reconnect...")
        
        for attempt in range(max_retries):
            try:
                # Stop current pipeline if it exists
                if self.pipeline:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                
                # Wait before reconnecting
                time.sleep(2)
                
                # Reinitialize camera
                self._initialize_camera()
                return True
                
            except Exception as e:
                print(f"✗ Reconnection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    
        print(f"✗ Failed to reconnect camera {self.serial_number} after {max_retries} attempts")
        return False

    def get_rgb_depth(self, max_retries=3):
        """Get RGB and depth images with automatic reconnection on failure"""
        for attempt in range(max_retries):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # 5 second timeout
                aligned_frames = self.align.process(frames)
                color = aligned_frames.get_color_frame()
                depth = aligned_frames.get_depth_frame()
                
                if not color or not depth:
                    raise RuntimeError("Failed to get valid color or depth frame")
                
                color_image = np.asanyarray(color.get_data())
                depth_image = np.asanyarray(depth.get_data()) / 1000.0  # Convert to meters
                
                # Convert BGR to RGB for consistency
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                return color_image, depth_image
                
            except (RuntimeError, Exception) as e:
                print(f"⚠ Error getting frames from camera {self.serial_number} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Try to reconnect the camera
                    if self._reconnect_camera():
                        print(f"✓ Camera {self.serial_number} reconnected, retrying capture...")
                        continue
                    else:
                        print(f"✗ Failed to reconnect camera {self.serial_number}")
                        break
                else:
                    print(f"✗ Failed to capture from camera {self.serial_number} after {max_retries} attempts")
                    raise

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
        """Safely stop the camera pipeline"""
        try:
            if self.pipeline:
                self.pipeline.stop()
                print(f"✓ Camera {self.serial_number} stopped successfully")
        except Exception as e:
            print(f"⚠ Warning stopping camera {self.serial_number}: {e}")

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
    """Create directory structure for saving data (matching original structure)"""
    dirs = {
        "rgb": os.path.join(HOME_DIR, scene_name, "img"),  # Match original 'img' naming
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
            "file_path": f"img/frame_{i+1:05d}.png",  # Match original 'img' naming
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
# Table Detection Function (adapted from original)
# ===============================

def detect_table_boundaries(camera, camera_pose):
    """
    Detect table boundaries using plane segmentation (adapted for RealSense)
    
    Args:
        camera: RealSense camera object
        camera_pose: Camera pose transform
        
    Returns:
        Table boundary coordinates and height
    """
    # Get RGB and depth data
    color, depth = camera.get_rgb_depth()
    pc = generate_pointcloud(depth, camera.intrinsics)
    
    # Transform to world frame
    pc_world = camera_pose * pc
    
    # Use Open3D to segment the plane (table)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_world.data.T)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    
    # Cluster points on the table plane
    table_points = pc_world.data.T[inliers]
    db = DBSCAN(eps=0.02, min_samples=20).fit(table_points)
    label_set = set(db.labels_)
    label_set.discard(-1)
    
    # Find the largest cluster and determine table boundaries
    max_size = 0
    x_min_world = y_min_world = z_min_world = 0
    x_max_world = y_max_world = z_max_world = 0
    
    for label in label_set:
        filtered_table_point_mask = db.labels_ == label
        filtered_table_pointcloud = table_points[filtered_table_point_mask]
        
        if len(filtered_table_pointcloud) > max_size:
            max_size = len(filtered_table_pointcloud)
            x_min_world = np.min(filtered_table_pointcloud[:, 0])
            x_max_world = np.max(filtered_table_pointcloud[:, 0])
            y_min_world = np.min(filtered_table_pointcloud[:, 1])
            y_max_world = np.max(filtered_table_pointcloud[:, 1])
            z_min_world = np.min(filtered_table_pointcloud[:, 2])
            z_max_world = np.max(filtered_table_pointcloud[:, 2])
    
    # Ensure min < max
    if x_min_world > x_max_world:
        x_min_world, x_max_world = x_max_world, x_min_world
    if y_min_world > y_max_world:
        y_min_world, y_max_world = y_max_world, y_min_world
    if z_min_world > z_max_world:
        z_min_world, z_max_world = z_max_world, z_min_world
    
    table_height = -plane_model[3]
    return x_min_world, x_max_world, y_min_world, y_max_world, z_min_world, z_max_world, table_height

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
    home_joints = np.array([0.07497743517160416, -2.0328524748431605, 1.277921199798584, -0.8172596136676233, -1.5602710882769983, 3.2171106338500977])
    robot.move_joint(home_joints, vel=1.0, acc=0.1)
    robot.gripper.open()
    print("✓ Robot initialized and moved to home position")
    
    # Initialize cameras
    print("Initializing cameras...")
    try:
        wrist_cam = RealSenseCamera(
            serial_number=camera_config['wrist_d435']['id'],
            width=640, height=480, fps=15, camera_name="d435"  # Reduced FPS
        )
        third_cam = RealSenseCamera(
            serial_number=camera_config['static_d405']['id'],
            width=640, height=480, fps=15, camera_name="d405"  # Reduced FPS
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
    
    # Detect table boundaries using wrist camera
    print("Detecting table boundaries...")
    wrist_pose = robot.get_pose()
    wrist_pose.from_frame = "wrist"
    wrist_pose.to_frame = "world"
    cam_pose = wrist_pose * wrist_to_d435
    
    x_min_world, x_max_world, y_min_world, y_max_world, z_min_world, z_max_world, table_height = detect_table_boundaries(wrist_cam, cam_pose)
    print(f"✓ Table boundaries detected: x=[{x_min_world:.3f}, {x_max_world:.3f}], y=[{y_min_world:.3f}, {y_max_world:.3f}], z=[{z_min_world:.3f}, {z_max_world:.3f}]")
    
    # Load trajectory (Note: using calibration_trajectory.npy instead of prime_centered_trajectory.npy from original)
    trajectory_path = os.path.join(calibration_save_path, "prime_centered_trajectory.npy")
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
        time.sleep(1.5)  # Increased wait time for better stabilization and camera cooling
        
        # Get current robot pose
        wrist_pose = robot.get_pose()
        wrist_pose.from_frame = "wrist"
        wrist_pose.to_frame = "world"
        
        # Calculate camera pose
        cam_pose = wrist_pose * wrist_to_d435
        
        # Capture from wrist camera with retry mechanism
        try:
            color_d435, depth_d435 = wrist_cam.get_rgb_depth()
        except Exception as e:
            print(f"✗ Critical error capturing frame {i+2}: {e}")
            print("Stopping capture to prevent further issues...")
            break
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
    
    # Generate and save combined point cloud with table-based filtering
    print("Generating combined point cloud...")
    
    # Filter out empty arrays before stacking
    valid_pc_arrays = [pc for pc in global_pc if len(pc) > 0]
    valid_rgb_arrays = [rgb for rgb in global_rgb if len(rgb) > 0]
    
    if len(valid_pc_arrays) > 0:
        pc_all = np.vstack(valid_pc_arrays)
        rgb_all = np.vstack(valid_rgb_arrays)
        
        # Filter point cloud based on table boundaries (similar to original)
        close_mask = (
            (pc_all[:, 0] >= x_min_world) & (pc_all[:, 0] <= x_max_world) &
            (pc_all[:, 1] >= y_min_world) & (pc_all[:, 1] <= y_max_world) &
            (pc_all[:, 2] >= z_min_world) & (pc_all[:, 2] <= z_max_world)
        )
        
        close_pointcloud = pc_all[close_mask]
        close_rgbcloud = rgb_all[close_mask]
        not_close_pointcloud = pc_all[~close_mask]
        not_close_rgbcloud = rgb_all[~close_mask]
        
        # Subsample with table-aware strategy
        num_gaussians_initialization = 100000  # Reduced from original 200k for efficiency
        
        if len(close_pointcloud) > 0:
            # Sample more densely from table region (foreground)
            close_samples = min(len(close_pointcloud), int(num_gaussians_initialization * 0.7))
            close_indices = np.random.choice(len(close_pointcloud), close_samples, replace=False)
            subsampled_close_pc = close_pointcloud[close_indices]
            subsampled_close_rgb = close_rgbcloud[close_indices]
            
            # Remove outliers using DBSCAN
            db = DBSCAN(eps=0.005, min_samples=20)
            labels = db.fit_predict(subsampled_close_pc)
            subsampled_close_pc = subsampled_close_pc[labels != -1]
            subsampled_close_rgb = subsampled_close_rgb[labels != -1]
        else:
            subsampled_close_pc = np.empty((0, 3))
            subsampled_close_rgb = np.empty((0, 3))
        
        if len(not_close_pointcloud) > 0:
            # Sample less densely from background
            bg_samples = min(len(not_close_pointcloud), 
                           int(num_gaussians_initialization * 0.3))
            bg_indices = np.random.choice(len(not_close_pointcloud), bg_samples, replace=False)
            subsampled_bg_pc = not_close_pointcloud[bg_indices]
            subsampled_bg_rgb = not_close_rgbcloud[bg_indices]
        else:
            subsampled_bg_pc = np.empty((0, 3))
            subsampled_bg_rgb = np.empty((0, 3))
        
        # Combine foreground and background
        if len(subsampled_close_pc) > 0 and len(subsampled_bg_pc) > 0:
            pc_all = np.vstack((subsampled_close_pc, subsampled_bg_pc))
            rgb_all = np.vstack((subsampled_close_rgb, subsampled_bg_rgb))
        elif len(subsampled_close_pc) > 0:
            pc_all = subsampled_close_pc
            rgb_all = subsampled_close_rgb
        elif len(subsampled_bg_pc) > 0:
            pc_all = subsampled_bg_pc
            rgb_all = subsampled_bg_rgb
        else:
            pc_all = np.empty((0, 3))
            rgb_all = np.empty((0, 3))
            
        print(f"✓ Point cloud filtered: {len(close_pointcloud)} foreground, {len(not_close_pointcloud)} background points")
        print(f"✓ Final point cloud: {len(pc_all)} points after subsampling")
        
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
    
    # Save table boundaries for later use (similar to original)
    table_bounding_cube = {
        'x_min': float(x_min_world),
        'x_max': float(x_max_world),
        'y_min': float(y_min_world),
        'y_max': float(y_max_world),
        'z_min': float(z_min_world),
        'z_max': float(z_max_world),
        'table_height': float(table_height)
    }
    with open(os.path.join(save_dirs["scene"], "table_bounding_cube.json"), 'w') as f:
        json.dump(table_bounding_cube, f, indent=4)
    print(f"✓ Table boundaries saved to table_bounding_cube.json")
    
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