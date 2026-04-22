#!/usr/bin/env python3
"""Test grid_sample_3d offset requirement."""
import torch
torch.manual_seed(42)

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import warnings
warnings.filterwarnings('ignore')

from flex_gemm.ops.grid_sample import grid_sample_3d

# Create a simple sparse grid
N = 8
C = 3
feats = torch.zeros(N, C, device='cuda', dtype=torch.float32)
coords = torch.zeros(N, 4, device='cuda', dtype=torch.int32)  # [batch, x, y, z]

for i in range(8):
    coords[i, 0] = 0  # batch
    coords[i, 1] = (i >> 2) & 1  # x
    coords[i, 2] = (i >> 1) & 1  # y
    coords[i, 3] = i & 1         # z
    feats[i, 0] = float(i) / 7.0
    feats[i, 1] = float(i) / 14.0
    feats[i, 2] = float(i) / 21.0

shape = torch.Size([1, C, 2, 2, 2])

print("Testing coordinate convention for grid_sample_3d:")
print("=" * 60)

# Test at voxel corners (integer coords) - should return 0 or interpolated
query_corner = torch.tensor([[[0.0, 0.0, 0.0]]], device='cuda', dtype=torch.float32)
result_corner = grid_sample_3d(feats, coords, shape, query_corner, mode='trilinear')
print(f"Query at (0,0,0) - corner: {result_corner[0,0].tolist()}")

# Test at voxel centers (integer + 0.5) - should return voxel value
query_center = torch.tensor([[[0.5, 0.5, 0.5]]], device='cuda', dtype=torch.float32)
result_center = grid_sample_3d(feats, coords, shape, query_center, mode='trilinear')
print(f"Query at (0.5, 0.5, 0.5) - center of voxel 0: {result_center[0,0].tolist()}")

print() 
print("Expected:")
print(f"  Voxel 0 attrs: {feats[0].tolist()}")
print() 

# Test another voxel
query_v1 = torch.tensor([[[0.5, 0.5, 1.5]]], device='cuda', dtype=torch.float32)
result_v1 = grid_sample_3d(feats, coords, shape, query_v1, mode='trilinear')
print(f"Query at (0.5, 0.5, 1.5) - center of voxel 1: {result_v1[0,0].tolist()}")
print(f"Expected voxel 1 attrs: {feats[1].tolist()}")

print()
print("CONCLUSION:")
print("  - grid_sample_3d uses (i+0.5) as voxel center coordinates")
print("  - To sample at voxel (x, y, z), use query (x+0.5, y+0.5, z+0.5)")
print("  - Interpolated positions need +0.5 offset to align with voxel centers")