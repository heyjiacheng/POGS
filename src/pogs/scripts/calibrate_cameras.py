# === Imports ===
import time
import numpy as np
import cv2
import os
import yaml
import pyrealsense2 as rs
from ur5py.ur5 import UR5Robot
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pathlib

# === Set paths for saving calibration data ===
file_path = os.path.dirname(os.path.realpath(__file__))  # Current script path
calibration_save_path = os.path.join(file_path, '../calibration_outputs')  # Save directory for calibration outputs
config_filepath = os.path.join(file_path, '../configs/camera_config.yaml')  # Config file path

# Create output directory if it doesn't exist
if not os.path.exists(calibration_save_path):
    os.makedirs(calibration_save_path)

# === Convert OpenCV rotation+translation to autolab RigidTransform ===
def rvec_tvec_to_transform(rvec, tvec, to_frame):
    if rvec is None or tvec is None:
        return None
    R_mat = cv2.Rodrigues(rvec)[0]  # Convert rotation vector to matrix
    return RigidTransform(rotation=R_mat, translation=tvec, from_frame="tag", to_frame=to_frame)

# === Clear the UR5 tool center point ===
def clear_tcp(robot):
    tcp = RigidTransform(translation=np.array([0, 0, 0]), from_frame='tool', to_frame='wrist')
    robot.set_tcp(tcp)

# === Detect ArUco marker and estimate its pose ===
def pose_estimation(frame, K, dist, tag_length, visualize=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)  # Use ARUCO_ORIGINAL dictionary
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)  # Detect markers

    if len(corners) == 0:
        if visualize:
            print("No markers found")
        return None

    # 3D points of the marker corners in marker frame
    obj_points = np.array([
        [-tag_length / 2, tag_length / 2, 0],
        [ tag_length / 2, tag_length / 2, 0],
        [ tag_length / 2, -tag_length / 2, 0],
        [-tag_length / 2, -tag_length / 2, 0]
    ], dtype=np.float32)

    for i in range(len(corners)):
        img_points = corners[i].reshape((4, 2))
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)  # Estimate pose
        if success:
            if visualize:
                frame_vis = cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.1)  # Draw coordinate axes
                cv2.imshow('aruco', frame_vis)
                cv2.waitKey(0)
            return frame, rvec, tvec
    return None

# === Real-time camera preview with ArUco detection ===
def show_camera_preview(d435, d405, tag_len):
    """
    Show real-time preview of both cameras with ArUco detection overlay
    Press 'q' to quit, 'c' to capture current pose
    Returns True if user pressed 'c' to capture, False if 'q' to quit
    """
    d = np.zeros(5)  # Assume no distortion
    
    # Define 3D points of the marker corners in marker frame (used by both cameras)
    obj_points = np.array([
        [-tag_len / 2, tag_len / 2, 0],
        [ tag_len / 2, tag_len / 2, 0],
        [ tag_len / 2, -tag_len / 2, 0],
        [-tag_len / 2, -tag_len / 2, 0]
    ], dtype=np.float32)
    
    while True:
        # Get frames from both cameras
        d435_img = d435.get_frame()
        d405_img = d405.get_frame()
        
        # Get camera intrinsics
        K_d435 = d435.get_intrinsics_matrix()
        K_d405 = d405.get_intrinsics_matrix()
        
        # Create display images
        display_d435 = d435_img.copy()
        display_d405 = d405_img.copy()
        
        # Detect ArUco markers and draw them
        gray_d435 = cv2.cvtColor(d435_img, cv2.COLOR_BGR2GRAY)
        gray_d405 = cv2.cvtColor(d405_img, cv2.COLOR_BGR2GRAY)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        
        # D435 detection and visualization
        corners_d435, ids_d435, _ = detector.detectMarkers(gray_d435)
        if len(corners_d435) > 0:
            cv2.aruco.drawDetectedMarkers(display_d435, corners_d435, ids_d435)
            # Try to estimate pose and draw axes
            for i in range(len(corners_d435)):
                img_points = corners_d435[i].reshape((4, 2))
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K_d435, d)
                if success:
                    display_d435 = cv2.drawFrameAxes(display_d435, K_d435, d, rvec, tvec, 0.05)
            
            # Add status text
            cv2.putText(display_d435, "ArUco DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_d435, "No ArUco detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # D405 detection and visualization  
        corners_d405, ids_d405, _ = detector.detectMarkers(gray_d405)
        if len(corners_d405) > 0:
            cv2.aruco.drawDetectedMarkers(display_d405, corners_d405, ids_d405)
            # Try to estimate pose and draw axes
            for i in range(len(corners_d405)):
                img_points = corners_d405[i].reshape((4, 2))
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K_d405, d)
                if success:
                    display_d405 = cv2.drawFrameAxes(display_d405, K_d405, d, rvec, tvec, 0.05)
            
            cv2.putText(display_d405, "ArUco DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_d405, "No ArUco detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(display_d435, "D435 (Wrist) - Press 'c' to capture, 'q' to quit", (10, display_d435.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_d405, "D405 (Static) - Press 'c' to capture, 'q' to quit", (10, display_d405.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Resize images for better display
        h1, w1 = display_d435.shape[:2]
        h2, w2 = display_d405.shape[:2]
        
        # Resize to same height for side-by-side display
        target_height = 480
        display_d435_resized = cv2.resize(display_d435, (int(w1 * target_height / h1), target_height))
        display_d405_resized = cv2.resize(display_d405, (int(w2 * target_height / h2), target_height))
        
        # Combine images side by side
        combined_img = np.hstack([display_d435_resized, display_d405_resized])
        
        # Show combined image
        cv2.imshow('Camera Preview - D435 (Left) | D405 (Right)', combined_img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        elif key == ord('c'):
            return True   # Capture
        elif key == 27:  # ESC key
            return False  # Quit

# === RealSense Camera Handler ===
class RealSenseCamera:
    def __init__(self, serial):
        self.pipeline = rs.pipeline()
        config = rs.config()
        print("serial", serial)
        config.enable_device(serial)  # Enable camera by serial number
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream
        profile = self.pipeline.start(config)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        for _ in range(5):
            self.pipeline.wait_for_frames()  # Skip warm-up frames
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def get_intrinsics_matrix(self):
        intr = self.intrinsics
        return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    def stop(self):
        self.pipeline.stop()

# === Main calibration procedure ===
def register_webcam():
    ur = UR5Robot(gripper=1)  # Initialize UR5
    time.sleep(1)
    ur.gripper.open()
    clear_tcp(ur)

    # Move robot to home position
    home_joints = np.array([0.11, -2.0, 1.2394, -0.75074, -1.64462, 3.29472])
    ur.move_joint(home_joints, vel=1.0, acc=0.1)

    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize RealSense cameras
    d435 = RealSenseCamera(serial=config['wrist_d435']['id'])
    d405 = RealSenseCamera(serial=config['static_d405']['id'])

    d = np.zeros(5)  # Assume no distortion
    tag_len = 0.10  # Tag size in meters

    # Initialize data collection arrays
    saved_joints = []
    world_to_wrist_list = []
    d435_to_tag_list = []
    d405_to_tag_list = []

    # Teaching mode settings - simplified approach
    teach_mode = True  # Enable teaching mode as requested
    
    # Trajectory management
    trajectory_path = pathlib.Path(calibration_save_path + "/prime_centered_trajectory.npy")
    automatic_path = False
    
    # Try to load pre-recorded trajectory if available
    if trajectory_path.exists() and input("Load existing trajectory? [y]/n: ").lower() != 'n':
        traj = np.load(trajectory_path)
        automatic_path = True
        teach_mode = False
        print(f"Loaded trajectory with {len(traj)} poses")
    else:
        # In teaching mode, ask for number of poses
        num_poses = input("How many poses do you want to collect? ")
        num_poses = int(num_poses)
        traj = list(range(num_poses))  # Just use indices
        print(f"Manual teaching mode: will collect {num_poses} poses")
        print("You will manually move the robot to each position")

    print("Starting calibration data collection...")
    
    for i in tqdm(range(len(traj))):
        if not automatic_path:
            # Manual teaching mode: let user physically move robot
            print(f"\nPose {i+1}/{len(traj)}")
            print("Manually move the robot arm to desired position...")
            print("Make sure the ArUco marker is visible to both cameras")
            print("Camera preview window will open - press 'c' to capture when ready, 'q' to quit")
            
            # Show camera preview and wait for user input
            capture_pose = show_camera_preview(d435, d405, tag_len)
            if not capture_pose:
                print("Calibration cancelled by user")
                d435.stop()
                d405.stop()
                cv2.destroyAllWindows()
                return
                
            time.sleep(0.2)  # Brief pause for stability
        else:
            # Automatic mode: move to pre-recorded pose
            ur.move_joint(traj[i], vel=1.0, acc=0.1)
            time.sleep(0.1)

        # Capture images from both cameras
        d435_img = d435.get_frame()
        d405_img = d405.get_frame()

        # Get current robot pose and joint positions
        H_rob_world = ur.get_pose()
        current_joints = ur.get_joints()
        print("Robot joints:", current_joints)
        
        # Get camera intrinsic matrices
        K_d435 = d435.get_intrinsics_matrix()
        K_d405 = d405.get_intrinsics_matrix()

        # Detect ArUco markers in both cameras
        out_d435 = None
        out_d405 = None
        
        if teach_mode:
            # In teaching mode, the preview should have shown markers, so direct detection
            out_d435 = pose_estimation(d435_img, K_d435, d, tag_len, False)
            out_d405 = pose_estimation(d405_img, K_d405, d, tag_len, False)
            
            if out_d435 is None:
                print("Warning: No marker detected in wrist camera after capture")
                print("Consider repositioning and trying again")
                continue
                
        else:
            # Automatic mode
            out_d435 = pose_estimation(d435_img, K_d435, d, tag_len, False)
            out_d405 = pose_estimation(d405_img, K_d405, d, tag_len, False)

        # Process wrist camera (D435) results
        if out_d435:
            _, rvec_d435, tvec_d435 = out_d435
            tf_d435_to_tag = rvec_tvec_to_transform(rvec_d435, tvec_d435, "d435")
            
            # Store transforms for calibration
            world_to_wrist = H_rob_world.as_frames("wrist", "world")
            world_to_wrist_list.append(world_to_wrist)
            d435_to_tag_list.append(tf_d435_to_tag)
            
            if teach_mode:
                saved_joints.append(current_joints)
            
            print("✓ D435 marker detected and processed")

        # Process static camera (D405) results - INDEPENDENTLY like original code
        if out_d405:
            _, rvec_d405, tvec_d405 = out_d405
            tf_d405_to_tag = rvec_tvec_to_transform(rvec_d405, tvec_d405, "d405")
            d405_to_tag_list.append(tf_d405_to_tag)
            print("✓ D405 marker detected and processed")
        
        # Report detection status
        if not out_d435 and not out_d405:
            print("✗ No markers detected in either camera - skipping this pose")
        elif not out_d435:
            print("✗ D435 marker not detected - skipping this pose for wrist calibration")
        elif not out_d405:
            print("⚠ D405 marker not detected - only wrist camera data collected")

    # Save trajectory if in teaching mode
    if teach_mode and saved_joints:
        np.save(trajectory_path, np.array(saved_joints))
        print(f"Saved trajectory with {len(saved_joints)} poses to {trajectory_path}")

    # Check if we have enough data for calibration
    if len(world_to_wrist_list) < 3:
        print("Error: Need at least 3 valid poses for calibration")
        d435.stop()
        d405.stop()
        return

    print(f"\nCollected {len(world_to_wrist_list)} valid poses for wrist camera calibration")
    print(f"Collected {len(d405_to_tag_list)} valid poses for static camera")

    # Estimate wrist camera to robot transform
    from pogs.camera.capture_utils import estimate_cam2rob
    
    # Convert transforms to required format for estimate_cam2rob
    H_chess_cams = [tf.as_frames("cb", "cam") for tf in d435_to_tag_list]
    H_rob_worlds = [tf.as_frames("rob", "world") for tf in world_to_wrist_list]
    
    H_cam_rob, _ = estimate_cam2rob(H_chess_cams, H_rob_worlds)

    print("\nEstimated wrist_to_d435:")
    print(H_cam_rob)
    
    # Save wrist camera calibration
    if "n" not in input("Save wrist camera calibration? [y]/n: "):
        H_cam_rob.to_frame = 'wrist'
        H_cam_rob.from_frame = 'd435'
        H_cam_rob.save(os.path.join(calibration_save_path, 'wrist_to_d435.tf'))
        print("Saved wrist_to_d435.tf")

    # Process static camera calibration if we have data
    if d405_to_tag_list and any(t is not None for t in d405_to_tag_list):
        # Filter out None values for processing
        valid_d405_data = [t for t in d405_to_tag_list if t is not None]
        print(f"\nProcessing static camera calibration with {len(valid_d405_data)} poses...")
        
        # Calculate average pose of tag as seen by static camera by averaging translations and rotations
        d405_to_tag_translations = [t.translation for t in valid_d405_data]
        d405_to_tag_rotations_euler = [R.from_matrix(t.rotation).as_euler('xyz') for t in valid_d405_data]

        avg_translation = np.mean(d405_to_tag_translations, axis=0)
        avg_rotation_euler = np.mean(d405_to_tag_rotations_euler, axis=0)
        avg_rotation_matrix = R.from_euler('xyz', avg_rotation_euler).as_matrix()

        d405_to_tag_avg = RigidTransform(
            rotation=avg_rotation_matrix,
            translation=avg_translation,
            from_frame="tag",
            to_frame="d405"
        )
        print("Average D405 to tag transform:")
        print(d405_to_tag_avg)
        
        # Calculate world to static camera transforms using INDIVIDUAL observations (like original code)
        world_to_d405_rvecs = []
        world_to_d405_tvecs = []
        
        for world_to_wrist, d435_to_tag, d405_to_tag in zip(world_to_wrist_list, d435_to_tag_list, d405_to_tag_list):
            if d405_to_tag is not None:  # Only process if we have valid static camera data
                # Transform chain: world -> wrist -> d435 -> tag -> d405 (using INDIVIDUAL d405_to_tag)
                world_to_d405 = world_to_wrist * H_cam_rob * d435_to_tag * d405_to_tag.inverse()
                
                # Convert to rotation vector and translation for averaging (like original code)
                world_to_d405_rvec, _ = cv2.Rodrigues(world_to_d405.rotation)
                world_to_d405_tvec = world_to_d405.translation
                
                # Store results
                world_to_d405_rvecs.append(world_to_d405_rvec)
                world_to_d405_tvecs.append(world_to_d405_tvec)

        # Calculate average translation and rotation for final transform (like original code)
        avg_world_to_d405_translation = np.mean(np.array(world_to_d405_tvecs), axis=0)
        avg_world_to_d405_rotation, _ = cv2.Rodrigues(np.mean(np.array(world_to_d405_rvecs), axis=0))
        
        world_to_d405 = RigidTransform(
            rotation=avg_world_to_d405_rotation,
            translation=avg_world_to_d405_translation,
            from_frame="d405",
            to_frame="world"
        )

        print("\nEstimated world_to_d405:")
        print(world_to_d405)
        
        # Save static camera calibration
        if "n" not in input("Save static camera calibration? [y]/n: "):
            world_to_d405.save(os.path.join(calibration_save_path, 'world_to_d405.tf'))
            print("Saved world_to_d405.tf")
    else:
        print("Warning: No static camera data collected - skipping static camera calibration")

    # Cleanup
    d435.stop()
    d405.stop()
    cv2.destroyAllWindows()
    print("\nCalibration completed!")

# === Script entrypoint ===
if __name__ == "__main__":
    register_webcam()