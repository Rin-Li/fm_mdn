from __future__ import annotations
import functools
import numpy as np
import open3d as o3d


def make_pcd(
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> o3d.geometry.PointCloud:
    points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3).astype(np.float64) / 255)
    pcd = o3d.geometry.PointCloud(points)
    pcd.colors = colors
    return pcd


def merge_pcds(
    voxel_size: float,
    n_points: int,
    pcds: list[o3d.geometry.PointCloud],
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox,
) -> o3d.geometry.PointCloud:
    merged_pcd = functools.reduce(lambda a, b: a + b, pcds, o3d.geometry.PointCloud())
    merged_pcd = merged_pcd.crop(ws_aabb)
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(downsampled_pcd.points) > n_points:
        ratio = n_points / len(downsampled_pcd.points)
        downsampled_pcd = downsampled_pcd.random_down_sample(ratio)
    if len(downsampled_pcd.points) < n_points:
        # Append zeros to make the point cloud have the desired number of points
        num_missing_points = n_points - len(downsampled_pcd.points)
        zeros = np.zeros((num_missing_points, 3))
        zeros_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zeros))
        zeros_pcd.colors = o3d.utility.Vector3dVector(zeros)
        downsampled_pcd += zeros_pcd
    return downsampled_pcd
