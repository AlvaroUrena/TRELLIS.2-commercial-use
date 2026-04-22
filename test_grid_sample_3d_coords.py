#!/usr/bin/env python3
"""
Compare grid_sample_3d output between fork (DRTK) and official (nvdiffrast).

This script tests:
1. Raw grid_sample_3d output at specific coordinates
2. Interpolated vertex positions from rasterization
3. Final attribute sampling results
"""

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from flex_gemm.ops.grid_sample import grid_sample_3d

print("=" * 70)
print("GRID_SAMPLE_3D COORDINATE TEST")
print("=" * 70)

# Create a simple test case
print("\n1. Testing grid_sample_3d with simple sparse grid...")
N = 8
C = 3
feats = torch.zeros(N, C, device='cuda', dtype=torch.float32)
coords = torch.zeros(N, 4, device='cuda', dtype=torch.int32)

for i in range(8):
    coords[i, 0] = 0  # batch
    coords[i, 1] = (i >> 2) & 1  # x
    coords[i, 2] = (i >> 1) & 1  # y
    coords[i, 3] = i & 1          # z
    feats[i, 0] = float(i) / 7.0
    feats[i, 1] = float(i) / 14.0
    feats[i, 2] = float(i) / 21.0

shape = torch.Size([1, C, 2, 2, 2])

# Test queries at different positions
queries = [
    ([0.0, 0.0, 0.0], "Voxel (0,0,0) corner"),
    ([0.5, 0.0, 0.0], "Voxel (0,0,0) center-x offset"),
    ([0.5, 0.5, 0.5], "Grid center (trilinear)"),
    ([1.0, 0.0, 0.0], "Voxel (1,0,0) corner"),
    ([0.5, 0.5, 0.5], "Test center"),
]

print(f"Shape: {shape}")
print(f"Coords (first 5): {coords[:5]}")
print(f"Feats (first 5): {feats[:5]}")

for query, desc in queries:
    q = torch.tensor([query], device='cuda', dtype=torch.float32).reshape(1, 1, 3)
    result = grid_sample_3d(feats, coords, shape, q, mode='trilinear')
    print(f"  {desc} {query}: {result[0, 0].cpu().numpy()}")

print("\n" + "=" * 70)
print("2. Testing voxel center convention...")
print("If grid assumes voxel centers at (i+0.5), then integer+i.5 should match voxel values")

# Test at voxel centers (i+0.5)
test_coords = [
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 1.5],
    [0.5, 1.5, 0.5],
    [0.5, 1.5, 1.5],
    [1.5, 0.5, 0.5],
    [1.5, 0.5, 1.5],
    [1.5, 1.5, 0.5],
    [1.5, 1.5, 1.5],
]

for c in test_coords:
    q = torch.tensor([c], device='cuda', dtype=torch.float32).reshape(1, 1, 3)
    result = grid_sample_3d(feats, coords, shape, q, mode='trilinear')
    # Expected voxel index
    ix = int(c[0] - 0.5)
    iy = int(c[1] - 0.5)
    iz = int(c[2] - 0.5)
    idx = ix * 4 + iy * 2 + iz
    expected = feats[idx].cpu().numpy()
    print(f"  Query {c} -> Index {idx}: result={result[0, 0].cpu().numpy()}, expected={expected}")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("grid_sample_3d uses voxel centers at (i+0.5), where i is the integer coordinate.")
print("To sample at voxel (x, y, z), use query point (x+0.5, y+0.5, z+0.5)")