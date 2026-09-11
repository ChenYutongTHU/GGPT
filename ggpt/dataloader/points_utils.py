from dataclasses import dataclass
import torch
import os
import numpy as np

@dataclass
class OctreeNode:
    center: torch.Tensor      # (3,)
    half_size: float
    indices: torch.Tensor     # point indices
    children: list = None


def _count_unique_voxels_1d(grid_xyz: torch.Tensor, side: int) -> int:
    """Count unique occupied voxels via 1D linear indices (faster than unique(..., dim=0))."""
    if grid_xyz.numel() == 0:
        return 0
    side = max(1, int(side))
    g = grid_xyz.to(torch.int64)
    lin = g[:, 0] * (side * side) + g[:, 1] * side + g[:, 2]
    return int(torch.unique(lin).shape[0])

def _half_size_le(half_size, bound: float) -> bool:
    hs = half_size.item() if isinstance(half_size, torch.Tensor) else float(half_size)
    return hs <= bound


def estimate_occupied_voxels(
    points,
    indices,
    center,
    half_size,
    grid_resolution: int,
    overlap_ratio: float = 0.0,
):
    if len(indices) == 0:
        return 0
    pts = points[indices]
    radius = float(half_size) * (1.0 + float(overlap_ratio))
    radius = max(radius, 1e-8)
    coord = (pts - center) / radius
    grid = (((coord.clamp(-1, 1) + 1.0) * 0.5) * int(grid_resolution)).floor().long()
    grid = grid.clamp(0, int(grid_resolution) - 1)
    return _count_unique_voxels_1d(grid, int(grid_resolution))


def estimate_multiscale_occupied_voxels(
    points,
    indices,
    center,
    half_size,
    grid_resolution: int,
    overlap_ratio: float,
    downsample_factors,
):
    if len(indices) == 0:
        return {int(s): 0 for s in downsample_factors}
    pts = points[indices]
    radius = float(half_size) * (1.0 + float(overlap_ratio))
    radius = max(radius, 1e-8)
    base_grid = (((((pts - center) / radius).clamp(-1, 1) + 1.0) * 0.5) * int(grid_resolution)).floor().long()
    base_grid = base_grid.clamp(0, int(grid_resolution) - 1)
    occ = {}
    base_side = int(grid_resolution)
    for s in downsample_factors:
        s = int(s)
        if s <= 1:
            g = base_grid
            side = base_side
        else:
            g = base_grid // s
            # Coordinates after floor-divide by s are in [0, ceil(base_side/s)-1].
            side = (base_side + s - 1) // s
        occ[s] = _count_unique_voxels_1d(g, side)
    return occ


def build_octree(points,
                 indices,
                 center,
                 half_size,
                 MAX,
                 min_half_size: float,
                 max_occupied_voxels_perchunk=None,
                 grid_resolution: int = 384,
                 overlap_ratio: float = 0.0,
                 occ_downsample_factors=(1,)):
    """
    points: (N, 3) tensor
    indices: (M,) tensor
    center: (3,)
    half_size: scalar tensor or float (cell half-extent, same units as points)
    min_half_size: stop splitting when cell half-extent <= this (min_radius_perchunk * scene_extent)
    """
    stop_by_size = _half_size_le(half_size, min_half_size)
    if stop_by_size:
        return OctreeNode(center, half_size, indices, children=None)
    stop_by_points = len(indices) <= MAX
    stop_by_occupancy = True
    if max_occupied_voxels_perchunk is not None:
        if isinstance(max_occupied_voxels_perchunk, dict):
            occupancy = estimate_multiscale_occupied_voxels(
                points,
                indices,
                center,
                half_size,
                grid_resolution=grid_resolution,
                overlap_ratio=overlap_ratio,
                downsample_factors=occ_downsample_factors,
            )
            for s, cap in max_occupied_voxels_perchunk.items():
                if occupancy.get(int(s), 0) > int(cap):
                    stop_by_occupancy = False
                    break
        else:
            occupancy = estimate_occupied_voxels(
                points,
                indices,
                center,
                half_size,
                grid_resolution=grid_resolution,
                overlap_ratio=overlap_ratio,
            )
            stop_by_occupancy = occupancy <= int(max_occupied_voxels_perchunk)
    if stop_by_points and stop_by_occupancy:
        return OctreeNode(center, half_size, indices, children=None)

    children = []
    quarter = half_size / 2.0
    pts = points[indices]
    # Assign each point to one octant once, instead of re-scanning for every child box.
    rel = pts - center
    octant_codes = (
        (rel[:, 0] >= 0).to(torch.int64)
        | ((rel[:, 1] >= 0).to(torch.int64) << 1)
        | ((rel[:, 2] >= 0).to(torch.int64) << 2)
    )

    for code in range(8):
        mask = octant_codes == code
        if not torch.any(mask):
            continue
        child_indices = indices[mask]
        child_center = center.clone()
        child_center[0] = center[0] + (quarter if (code & 1) else -quarter)
        child_center[1] = center[1] + (quarter if (code & 2) else -quarter)
        child_center[2] = center[2] + (quarter if (code & 4) else -quarter)
        child = build_octree(
            points,
            child_indices,
            child_center,
            quarter,
            MAX,
            min_half_size,
            max_occupied_voxels_perchunk=max_occupied_voxels_perchunk,
            grid_resolution=grid_resolution,
            overlap_ratio=overlap_ratio,
            occ_downsample_factors=occ_downsample_factors,
        )
        children.append(child)

    return OctreeNode(center, half_size, indices=None, children=children)


def make_root(points, trim_quantile: float = 0.0):
    tq = float(trim_quantile)
    if tq > 0.0:
        tq = min(max(tq, 0.0), 0.49)
        if points.shape[0] >= 8:
            lo = torch.quantile(points, tq, dim=0)
            hi = torch.quantile(points, 1.0 - tq, dim=0)
            min_xyz, max_xyz = lo, hi
        else:
            min_xyz = torch.min(points, dim=0)[0]
            max_xyz = torch.max(points, dim=0)[0]
    else:
        min_xyz = torch.min(points, dim=0)[0]
        max_xyz = torch.max(points, dim=0)[0]
    center = (min_xyz + max_xyz) / 2.0
    extent = max_xyz - min_xyz
    scene_extent = torch.max(extent)
    half_size = scene_extent / 2.0
    return center, half_size, scene_extent


def collect_leaves(node, leaves):
    if node.children is None:
        leaves.append(node)
    else:
        for c in node.children:
            collect_leaves(c, leaves)


def chunk_by_octree(
    points,
    MAX,
    min_radius_perchunk: float = 0.2,
    max_occupied_voxels_perchunk=None,
    grid_resolution: int = 384,
    overlap_ratio: float = 0.0,
    occ_downsample_factors=(1,),
    root_trim_quantile: float = 0.0,
):
    """
    min_radius_perchunk: octree stops subdividing when cell half-extent <= this fraction
    of scene_extent (max AABB edge length of points).
    """
    indices = torch.arange(points.shape[0], device=points.device)
    center, half_size, scene_extent = make_root(
        points, trim_quantile=root_trim_quantile
    )
    min_half_size = float(min_radius_perchunk) * float(scene_extent.item())
    root = build_octree(
        points,
        indices,
        center,
        half_size,
        MAX=MAX,
        min_half_size=min_half_size,
        max_occupied_voxels_perchunk=max_occupied_voxels_perchunk,
        grid_resolution=grid_resolution,
        overlap_ratio=overlap_ratio,
        occ_downsample_factors=occ_downsample_factors,
    )
    leaves = []
    collect_leaves(root, leaves)
    #chunks = [leaf.indices for leaf in leaves]
    #return chunks
    return leaves


def pca_transform(points, eigvecs=None, mean=None):
    reshape_flag = None
    if points.dim() != 2:
        reshape_flag = points.shape[:-1]
        points = points.reshape(-1, 3)
    
    if eigvecs is not None and mean is not None:
        points_pca = (points - mean) @ eigvecs
    else:
        mean = torch.mean(points, dim=0)
        X = points - mean
        C = (X.T @ X) / X.shape[0]  # covariance
        eigvals, eigvecs = torch.linalg.eigh(C)
        order = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, order]
        # deterministic sign fixing
        for i in range(3):
            j = torch.argmax(torch.abs(eigvecs[:, i]))
            if eigvecs[j, i] < 0:
                eigvecs[:, i] *= -1

        points_pca = X @ eigvecs
    if reshape_flag is not None:
        points_pca = points_pca.reshape(*reshape_flag, 3)
    return points_pca, eigvecs, mean




