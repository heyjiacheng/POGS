#!/usr/bin/env python3
"""
Automated Pipeline for Robot Vision and Action
Sequentially executes camera capture, vision processing, and robot action
"""

import os
import sys
import json
import argparse
import subprocess

class AutoPipeline:
    def __init__(self, execute_robot=False):
        self.execute_robot = execute_robot
        self.data_path = "./data/realsense_captures"
        self.script_dir = "./scripts"
        
    def run_camera_capture(self):
        """Run camera capture with live view and key press confirmation"""
        print("=== Step 1: Camera Capture ===")
        
        # Execute the camera script with modifications
        self._run_modified_camera()
        
    def _run_modified_camera(self):
        """Run modified camera capture with live view"""
        import pyrealsense2 as rs
        import numpy as np
        import cv2
        
        # Create save directory
        os.makedirs(self.data_path, exist_ok=True)
        
        # Configure streams
        pipeline = rs.pipeline()
        config = rs.config()
        
        target_serial = "819612070593"
        config.enable_device(target_serial)
        
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        pipeline.start(config)
        
        try:
            print("Camera started. Showing live view...")
            print("Press SPACE in the camera window to capture image, or press 'q' to quit")
            
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Apply colormap on depth image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Combine images for display
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
                
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                
                # Add instruction text on the image
                instruction_text = "Press SPACE to capture, 'q' to quit"
                cv2.putText(images, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show images
                cv2.namedWindow('RealSense Live View', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense Live View', images)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting without capture...")
                    break
                elif key == ord(' '):  # Space key to capture
                    # Save images
                    color_path = os.path.join(self.data_path, 'varied_camera_raw.png')  
                    depth_path = os.path.join(self.data_path, 'varied_camera_depth.npy')
                    
                    cv2.imwrite(color_path, color_image)
                    np.save(depth_path, depth_image)
                    
                    print(f"Images captured and saved to {self.data_path}")
                    print(f"- RGB: varied_camera_raw.png")
                    print(f"- Depth: varied_camera_depth.npy")
                    break
                
        finally:
            cv2.destroyAllWindows()
            pipeline.stop()
    
    def run_vision_processing(self):
        """Run vision processing"""
        print("\n=== Step 2: Vision Processing ===")
        
        # Default instruction
        instruction = "Drop the box cutter into the blue box."
        
        # Run vision processing as subprocess to avoid memory conflicts
        vision_script = os.path.join(self.script_dir, "real_vision.py")
        cmd = [
            sys.executable, vision_script,
            "--instruction", instruction,
            "--data_path", self.data_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Vision processing failed with error:")
            print(result.stderr)
            raise RuntimeError("Vision processing failed")
        
        print(result.stdout)
        
        # Extract rekep_program_dir from output
        rekep_program_dir = self._extract_rekep_dir_from_output(result.stdout)
        
        # Display images sequentially after vision processing is complete
        self._display_images_sequentially()
        
        return rekep_program_dir
    
    def _extract_rekep_dir_from_output(self, output):
        """Extract rekep program directory from vision script output"""
        lines = output.strip().split('\n')
        for line in lines:
            if 'rekep_program_dir:' in line:
                # Extract path from debug output
                return line.split('rekep_program_dir: ')[-1].strip()
        
        # Fallback: look for latest directory in vlm_query
        vlm_query_dir = "./vlm_query/"
        if os.path.exists(vlm_query_dir):
            vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                       if os.path.isdir(os.path.join(vlm_query_dir, d))]
            if vlm_dirs:
                return max(vlm_dirs, key=os.path.getmtime)
        
        return None
    
    def _display_images_sequentially(self):
        """Display three images sequentially with key press navigation"""
        import cv2
        import time
        
        print("\nWaiting for vision processing images to be generated...")
        
        # Try both .png and .jpg extensions and different locations
        image_files = [
            ('data/dinox_bbox', 'DINO-X Bounding Boxes'),
            ('data/dinox_mask', 'DINO-X Masks'),  
            ('data/rekep_with_keypoints', 'ReKep with Keypoints')
        ]
        
        # Wait a bit for files to be fully written
        time.sleep(2)
        
        for img_base, title in image_files:
            # Try different extensions and wait for file to exist
            img_path = None
            max_wait = 10  # seconds
            waited = 0
            
            while waited < max_wait:
                for ext in ['.png', '.jpg']:
                    test_path = img_base + ext
                    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                        img_path = test_path
                        break
                
                if img_path:
                    break
                    
                time.sleep(1)
                waited += 1
            
            if img_path:
                print(f"\nFound image: {img_path}")
                print(f"Displaying: {title} (press any key in the window to continue)")
                img = cv2.imread(img_path)

                if img is None or not img.any():
                    print(f"Warning: {img_path} appears to be empty or corrupted")
                    continue

                # Optional: limit resolution
                h, w = img.shape[:2]
                if h > 800 or w > 1200:
                    scale = min(800/h, 1200/w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))

                # Add instruction text on the image
                instruction_text = "Press any key to continue"
                cv2.putText(img, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow(title, img)
                cv2.waitKey(0)  # Wait for any key press in the window
                cv2.destroyAllWindows()
            else:
                print(f"Warning: No image found for {title} after waiting {max_wait}s")
    
    def run_robot_action(self, rekep_program_dir):
        """Run robot action processing"""
        print("\n=== Step 3: Robot Action Planning ===")
        
        if not rekep_program_dir:
            print("Error: No rekep program directory provided")
            return None
        
        instruction = "Drop the box cutter into the blue box."
        
        # Run robot action as subprocess
        action_script = os.path.join(self.script_dir, "real_action.py")
        cmd = [
            sys.executable, action_script,
            "--instruction", instruction,
            "--rekep_program_dir", rekep_program_dir
        ]
        
        if self.execute_robot:
            print("Executing robot actions...")
        else:
            print("Generating action plan (robot execution disabled)...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Robot action processing failed with error:")
            print(result.stderr)
            raise RuntimeError("Robot action processing failed")
        
        print(result.stdout)
        
        return self._load_action_sequence()
    
    def _load_action_sequence(self):
        """Load the generated all_subgoals"""
        all_subgoals_path = "./outputs/all_subgoals.json"
        
        if os.path.exists(all_subgoals_path):
            with open(all_subgoals_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def save_final_output(self, all_subgoals_data):
        """Display final output summary"""
        if all_subgoals_data:
            print(f"\n=== Final Output: all_subgoals.json ===")
            print(f"File location: ./outputs/all_subgoals.json")
            
            num_stages = all_subgoals_data.get('num_stages', 0)
            print(f"Total stages processed: {num_stages}")
            
            # Print each subgoal position
            for subgoal in all_subgoals_data.get('subgoals', []):
                stage = subgoal['stage']
                pos = subgoal['subgoal_pose'][:3]
                action = "GRASP" if subgoal['is_grasp_stage'] else ("RELEASE" if subgoal['is_release_stage'] else "MOVE")
                print(f"Stage {stage} ({action}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        else:
            print("Warning: No subgoals data found")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("Starting Automated Robot Pipeline")
        print(f"Robot execution: {'ENABLED' if self.execute_robot else 'DISABLED'}")
        
        try:
            # Step 1: Camera capture
            self.run_camera_capture()
            
            # Step 2: Vision processing
            rekep_program_dir = self.run_vision_processing()
            
            # Step 3: Robot action
            all_subgoals_data = self.run_robot_action(rekep_program_dir)
            
            # Step 4: Display final output summary
            self.save_final_output(all_subgoals_data)
            
            print("\n=== Pipeline Completed Successfully ===")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Automated Robot Pipeline')
    parser.add_argument('--execute-robot', action='store_true', 
                       help='Actually execute robot actions (default: plan only)')
    args = parser.parse_args()
    
    pipeline = AutoPipeline(execute_robot=args.execute_robot)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()