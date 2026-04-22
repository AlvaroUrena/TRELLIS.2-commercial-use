#!/usr/bin/env python3
"""
Debug projection matrix and clip coordinates.
"""

import torch
import numpy as np
from trellis2.utils import render_utils

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.
    return ret

print("=" * 70)
print("CHECKING INTRINSICS AND PROJECTION")
print("=" * 70)

extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

print("Extrinsics:")
print(extrinsics[0])
print("\nIntrinsics:")
print(intrinsics[0])
print()

resolution = 512
near = 1.0
far = 100.0

perspective = intrinsics_to_projection(intrinsics[0], near, far)
print("Perspective matrix:")
print(perspective)
print()

full_proj = perspective @ extrinsics[0]
print("Full projection matrix (perspective @ extrinsics):")
print(full_proj)
print()

# Test with unit cube vertex
test_vertices = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.1, 0.2, 0.3],
], device='cuda', dtype=torch.float32)

vertices_homo = torch.cat([test_vertices, torch.ones_like(test_vertices[..., :1])], dim=-1)
vertices_clip = vertices_homo @ full_proj.T

print("Test vertices (in object space):")
print(test_vertices)
print()

print("vertices_clip (after full projection):")
print(vertices_clip)
print("  X range:", vertices_clip[:, 0].min().item(), "-", vertices_clip[:, 0].max().item())
print("  Y range:", vertices_clip[:, 1].min().item(), "-", vertices_clip[:, 1].max().item())
print("  Z range:", vertices_clip[:, 2].min().item(), "-", vertices_clip[:, 2].max().item())
print("  W range:", vertices_clip[:, 3].min().item(), "-", vertices_clip[:, 3].max().item())
print()

# Compute pixel coordinates
w_clip = vertices_clip[..., 3].clamp(min=1e-8)
x_ndc = vertices_clip[..., 0] / w_clip
y_ndc = -vertices_clip[..., 1] / w_clip

h, w = resolution, resolution
x_pix = (x_ndc + 1) * 0.5 * w - 0.5
y_pix = (h - 1) - ((y_ndc + 1) * 0.5 * h - 0.5)

print("NDC coords:")
print("  x_ndc:", x_ndc)
print("  y_ndc:", y_ndc)
print()

print("Pixel coords:")
print("  x_pix:", x_pix)
print("  y_pix:", y_pix)
print()

# Check if pixel coords are in reasonable range
print("Pixel coords in image bounds?")
print("  x in [0, 511]:", (x_pix >= 0).all().item() and (x_pix < 512).all().item())
print("  y in [0, 511]:", (y_pix >= 0).all().item() and (y_pix < 512).all().item())