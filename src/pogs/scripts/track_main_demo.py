import torch
import viser  # For 3D scene visualization
import viser.transforms as vtf  # For SE3 transformations
import time
import numpy as np
import tyro  # Command-line interface for function arguments
from pathlib import Path
from autolab_core import RigidTransform  # For rigid transforms
from pogs.tracking.realsense import RealSense  # You must create/adapt this like Zed.py
from pogs.tracking.optim import Optimizer  # Core optimization pipeline
import warp as wp
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
import os
import cv2

# ============================================================================
# Path Setup and Constants
# ============================================================================

# Get the path of the current script file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load static camera pose (D405 in place of ZED2) — assumed calibrated
WORLD_TO_D405 = RigidTransform.load(
    os.path.join(dir_path, "../tracking/data/calibration_outputs/world_to_d405.tf")
)

# Set the compute device to GPU
DEVICE = 'cuda:0'


# ============================================================================
# Main Function: Tracking Demo with RealSense
# ============================================================================

def main(
    config_path: str = "/home/yujustin/pogs/outputs/drill_realsense_d435/pogs/2025-02-07_004727/config.yml",
    offline_folder: str = '/home/yujustin/pogs/data/demonstrations/drill_realsense_d435'
):
    """Tracking demo using wrist-mounted RealSense D435 and static D405."""
    
    # Load image and depth data paths
    image_folder = os.path.join(offline_folder, "color")
    depth_folder = os.path.join(offline_folder, "depth")
    image_paths = sorted(os.listdir(image_folder))
    depth_paths = sorted(os.listdir(depth_folder))
    
    # Start a visualization server (Viser for 3D viewing)
    server = viser.ViserServer()

    # Initialize warp (physics or differentiable rendering library)
    wp.init()

    # Create the OpenCLIP encoder used for semantic/text-based matching
    clip_encoder = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16", 
        clip_model_pretrained="laion2b_s34b_b88k", 
        clip_n_dims=512, 
        device=DEVICE
    ).setup()
    assert isinstance(clip_encoder, OpenCLIPNetwork)

    # Load static camera transform for D405 (previously calibrated)
    camera_tf = WORLD_TO_D405

    # Visualize static camera frame
    camera_frame = server.add_frame(
        "camera",
        position=camera_tf.translation,
        wxyz=camera_tf.quaternion,
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.005,
    )

    # Load the first image and depth frame for initialization
    initial_image_path = os.path.join(image_folder, image_paths[0])
    initial_depth_path = os.path.join(depth_folder, depth_paths[0])
    img_numpy = cv2.imread(initial_image_path)
    depth_numpy = np.load(initial_depth_path)

    # Convert image to RGB torch tensor
    l = torch.from_numpy(cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(DEVICE)
    depth = torch.from_numpy(depth_numpy).float().to(DEVICE)

    # RealSense D435 intrinsics (replace with your actual calibrated intrinsics if needed)
    realsense_K = np.array([
        [616.36529541, 0.0, 310.25881958],
        [0.0, 616.20294189, 236.59980774],
        [0.0, 0.0, 1.0]
    ])

    # Initialize optimizer with config path and camera intrinsics
    toad_opt = Optimizer(
        Path(config_path),
        realsense_K,
        l.shape[2],  # width
        l.shape[1],  # height
        init_cam_pose=torch.from_numpy(
            vtf.SE3(wxyz_xyz=np.array([*camera_frame.wxyz, *camera_frame.position])).as_matrix()[None, :3, :]
        ).float(),
    )

    # Lists to store tracking results and visualizations
    real_frames = []
    rendered_rgb_frames = []
    part_deltas = []
    save_videos = True
    obj_label_list = [None for _ in range(toad_opt.num_groups)]

    # Initial frame setup
    toad_opt.set_frame(l, toad_opt.cam2world_ns_ds, depth)
    toad_opt.init_obj_pose()

    print("Starting main tracking loop")

    while not toad_opt.initialized:
        time.sleep(0.1)

    for img_path, depth_path in zip(image_paths, depth_paths):
        img_numpy = cv2.imread(os.path.join(image_folder, img_path))
        depth_numpy = np.load(os.path.join(depth_folder, depth_path))

        left = torch.from_numpy(cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(DEVICE)
        depth = torch.from_numpy(depth_numpy).float().to(DEVICE)

        # Feed image and depth to optimizer
        toad_opt.set_observation(left, toad_opt.cam2world_ns, depth)
        outputs = toad_opt.step_opt(niter=15)

        # Add bounding boxes to real image
        rgb_img = left.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        for i, frame in enumerate(toad_opt.optimizer.frame.roi_frames):
            rgb_img = cv2.rectangle(rgb_img, (frame.xmin, frame.ymin), (frame.xmax, frame.ymax), (255, 0, 0), 2)

        # Display raw and rendered images
        server.scene.add_image("cam/realsense_d435", rgb_img, render_width=rgb_img.shape[1]/2500,
                               render_height=rgb_img.shape[0]/2500, position=(-0.5, -0.5, 0.5),
                               wxyz=(0, -1, 0, 0), visible=True)

        server.scene.add_image("cam/rendered", outputs["rgb"].cpu().numpy(),
                               render_width=outputs["rgb"].shape[1]/2500,
                               render_height=outputs["rgb"].shape[0]/2500,
                               position=(0.5, -0.5, 0.5), wxyz=(0, -1, 0, 0), visible=True)

        if save_videos:
            real_frames.append(rgb_img)
            rendered_rgb_frames.append(outputs["rgb"].cpu().numpy())

        # Update object part poses
        tf_list = toad_opt.get_parts2world()
        part_deltas.append(tf_list)
        for idx, tf in enumerate(tf_list):
            server.add_frame(f"object/group_{idx}", position=tf.translation(), wxyz=tf.rotation().wxyz,
                             show_axes=True, axes_length=0.05, axes_radius=0.001)
            mesh = toad_opt.toad_object.meshes[idx]
            server.add_mesh_trimesh(f"object/group_{idx}/mesh", mesh=mesh)

            if idx == toad_opt.max_relevancy_label:
                obj_label_list[idx] = server.add_label(
                    f"object/group_{idx}/label", text=toad_opt.max_relevancy_text, position=(0, 0, 0.05)
                )
            else:
                if obj_label_list[idx] is not None:
                    obj_label_list[idx].remove()

        # Visualize 3D point cloud from depth
        K = torch.from_numpy(realsense_K).float().cuda()
        points, colors = RealSense.project_depth(left, depth, K, depth_threshold=1.0, subsample=6)
        server.add_point_cloud("camera/points", points=points, colors=colors, point_size=0.001)


if __name__ == "__main__":
    tyro.cli(main)