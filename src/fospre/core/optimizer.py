"""
POGS optimizer setup and management.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple

from pogs.tracking.optim import Optimizer
from pogs.encoders.openclip_encoder import OpenCLIPNetworkConfig
from .utils import safe_sleep


def setup_optimizer(config_path: Path) -> Optimizer:
    """Initialize the POGS optimizer.
    
    Args:
        config_path: Path to the POGS config file
        
    Returns:
        Initialized Optimizer instance
    """
    # Get dataset to extract camera parameters
    from nerfstudio.utils.eval_utils import eval_setup
    _, temp_pipeline, _, _ = eval_setup(config_path)
    dataset = temp_pipeline.datamanager.train_dataset
    
    if dataset is not None and len(dataset) > 0:
        ref_camera = dataset.cameras[0]
        width, height = int(ref_camera.width.item()), int(ref_camera.height.item())
        fx, fy = ref_camera.fx.item(), ref_camera.fy.item()
        cx, cy = ref_camera.cx.item(), ref_camera.cy.item()
    else:
        width, height = 640, 480
        fx = fy = 500.0
        cx, cy = width / 2.0, height / 2.0
    
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dummy_cam_pose = torch.from_numpy(np.eye(4)[:3, :]).float()[None, :]
    
    optimizer = Optimizer(config_path, K, width, height, init_cam_pose=dummy_cam_pose)
    del temp_pipeline
    safe_sleep(2)  # Wait for optimizer to load
    
    return optimizer


def find_target_object(optimizer: Optimizer, semantic_query: str) -> int:
    """Find object index matching the semantic query.
    
    Args:
        optimizer: POGS optimizer instance
        semantic_query: Semantic description of the target object
        
    Returns:
        Index of the target object
    """
    clip_encoder = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16", 
        clip_model_pretrained="laion2b_s34b_b88k", 
        clip_n_dims=512, 
        device='cuda:0'
    ).setup()
    
    clip_encoder.set_positives([semantic_query])
    relevancy = optimizer.get_clip_relevancy(clip_encoder)
    group_masks = optimizer.optimizer.group_masks
    
    relevancy_avg = [torch.mean(relevancy[:, 0:1][mask]) for mask in group_masks]
    target_idx = torch.argmax(torch.tensor(relevancy_avg)).item()
    
    print(f"Found object '{semantic_query}' at index {target_idx}")
    return target_idx