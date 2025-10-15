#!/usr/bin/env python3
"""
Automatically cluster a trained POGS scene, locate the target object via language,
and export the resulting crop metadata to clusters.npy for downstream pipelines.

This replaces the manual viewer actions: Cluster Scene → language query on "box cutter"
→ Crop to Click → Add Crop to Group List.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import tyro
from scipy.spatial.transform import Rotation as R

from fospre.core.utils import find_latest_config
from nerfstudio.utils.eval_utils import eval_setup
from pogs.field_components.gaussian_fieldheadnames import GaussianFieldHeadNames
from pogs.pogs import POGSModel
from pogs.pogs_pipeline import POGSPipeline


def _ensure_model_ready(pipeline: POGSPipeline) -> None:
    """Render once to populate fields such as best_scales."""
    dataset = pipeline.datamanager.train_dataset
    if dataset is None or len(dataset.cameras) == 0:
        raise RuntimeError("Training dataset is empty; cannot bootstrap model outputs.")

    device = pipeline.model.gauss_params["means"].device
    camera = dataset.cameras[0:1].to(device)
    with torch.no_grad():
        pipeline.model.get_outputs(camera, tracking=False, BLOCK_WIDTH=16, rgb_only=False)

    if getattr(pipeline.model, "best_scales", None) is None:
        raise RuntimeError("Model best_scales not populated after render; aborting.")


def _compute_relevancy(model: POGSModel, query: str) -> torch.Tensor:
    """Compute CLIP relevancy per Gaussian for the provided language query."""
    device = model.gauss_params["means"].device
    model.image_encoder.set_positives([query])

    means = model.gauss_params["means"].detach()
    distances, indices = model.k_nearest_sklearn(means, 3, include_self=True)
    indices_t = torch.from_numpy(indices).to(device).view(-1)

    opacities = model.gauss_params["opacities"][indices_t].view(-1, 4)
    weights = F.softmax(torch.sigmoid(opacities), dim=-1)

    points = means[indices_t]
    clip_hash_encoding = model.gaussian_field.get_hash(points)
    clip_hash_encoding = clip_hash_encoding.view(-1, 4, clip_hash_encoding.shape[1])
    clip_hash_encoding = (clip_hash_encoding * weights.unsqueeze(-1)).sum(dim=1)

    scales = model.best_scales[0].to(device) * torch.ones(model.num_points, 1, device=device)
    clip_feats = model.gaussian_field.get_outputs_from_feature(
        clip_hash_encoding, scales
    )[GaussianFieldHeadNames.CLIP].to(dtype=torch.float32)

    normalized = clip_feats / (clip_feats.norm(dim=-1, keepdim=True) + 1e-6)
    relevancy = model.image_encoder.get_relevancy(normalized, 0).view(model.num_points, -1)
    return relevancy[:, 0]


def _compute_transform(cluster_points: np.ndarray) -> np.ndarray:
    """Estimate crop transform (wxyz + translation) for a cluster."""
    centroid = cluster_points.mean(axis=0)
    orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if cluster_points.shape[0] >= 3:
        try:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_points))
            obb = pcd.get_oriented_bounding_box()
            R_mat = np.asarray(obb.R)

            target_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            dot_products = np.abs(R_mat.T @ target_z)
            z_index = int(np.argmax(dot_products))
            remaining = [i for i in range(3) if i != z_index]

            new_x = R_mat[:, remaining[0]]
            if np.linalg.norm(new_x) < 1e-6:
                raise ValueError("Degenerate axis for crop orientation.")
            new_x = new_x / np.linalg.norm(new_x)

            new_y = np.cross(target_z, new_x)
            if np.linalg.norm(new_y) < 1e-6:
                raise ValueError("Degenerate cross-product for crop orientation.")
            new_y = new_y / np.linalg.norm(new_y)

            R_new = np.column_stack((new_x, new_y, target_z))
            quat_xyzw = R.from_matrix(R_new).as_quat()
            orientation = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
            )
        except Exception:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    transform = np.concatenate([orientation, centroid.astype(np.float32)])
    return transform


@dataclass
class Args:
    config_path: Optional[Path] = None
    query: str = "apple"
    cluster_eps: float = 0.012
    output_path: Optional[Path] = None


def main(args: Args) -> None:
    config_path = args.config_path or find_latest_config()
    config_path = config_path.resolve()

    train_config, pipeline, _, _ = eval_setup(config_path)
    assert isinstance(pipeline, POGSPipeline)

    pipeline.model.eval()
    _ensure_model_ready(pipeline)

    labels = pipeline.model.cluster(args.cluster_eps)
    if labels is None:
        raise RuntimeError("Clustering returned None; check cluster_eps or model state.")

    cluster_labels = torch.from_numpy(labels.astype(np.int64))
    pipeline.model.cluster_labels = cluster_labels

    relevancy = _compute_relevancy(pipeline.model, args.query)
    positive_mask = cluster_labels >= 0
    valid_ids = torch.unique(cluster_labels[positive_mask])
    if len(valid_ids) == 0:
        raise RuntimeError("No valid clusters found after filtering.")

    best_cluster = max(
        valid_ids.tolist(),
        key=lambda cid: relevancy[cluster_labels == cid].mean().item(),
    )

    keep_inds = torch.nonzero(cluster_labels == best_cluster, as_tuple=False).squeeze(-1)
    keep_inds = keep_inds.sort().values.cpu()
    pipeline.model.keep_inds = keep_inds

    means_cpu = pipeline.model.gauss_params["means"].detach().cpu()
    cluster_points = means_cpu[keep_inds.numpy()].numpy()
    transform = _compute_transform(cluster_points)
    cgtf_stack = np.stack([transform])
    pipeline.cgtf_stack = cgtf_stack
    pipeline.model.cgtf_stack = cgtf_stack

    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else config_path.parent.parent.parent / "clustered_object.ply"
    )

    # Create Open3D point cloud and save as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)

    # Optionally add colors if available in the model
    try:
        features_dc = pipeline.model.gauss_params["features_dc"].detach().cpu().numpy()
        # Convert from spherical harmonics to RGB (SH degree 0 is just a constant)
        # features_dc is already in RGB space after conversion from SH
        C0 = 0.28209479177387814  # SH constant for degree 0
        colors = (features_dc[keep_inds.numpy()] * C0 + 0.5).clip(0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception as e:
        print(f"Warning: Could not extract colors: {e}")

    o3d.io.write_point_cloud(str(output_path), pcd)

    print(f"Saved clustered point cloud to {output_path}")
    print(f"Selected cluster id: {best_cluster} (size={keep_inds.numel()})")
    print(f"Cluster centroid: {cluster_points.mean(axis=0)}")
    print(f"Cluster bounds: min={cluster_points.min(axis=0)}, max={cluster_points.max(axis=0)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
