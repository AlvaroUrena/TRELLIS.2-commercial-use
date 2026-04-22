#!/usr/bin/env python3
"""
Test grid_sample_3d with actual mesh data to find why it returns zeros.
"""

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from flex_gemm.ops.grid_sample import grid_sample_3d
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image

print("=" * 70)
print("GRID_SAMPLE_3D FULL MESH TEST")
print("=" * 70)

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh.simplify(16777216)

print(f"mesh.coords shape: {mesh.coords.shape}")
print(f"mesh.attrs shape: {mesh.attrs.shape}")
print(f"voxel_shape: {mesh.voxel_shape}")
print(f"origin: {mesh.origin}")
print(f"voxel_size: {mesh.voxel_size}")
print()

coords_with_batch = torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1)
print(f"coords_with_batch shape: {coords_with_batch.shape}")
print(f"coords_with_batch[:5]:")
print(coords_with_batch[:5])
print()

print(f"attrs sample (first 5):")
print(mesh.attrs[:5])
print()

xyz_voxel_test = torch.tensor([
    [14.5, 239.5, 363.5],
    [15.5, 239.5, 363.5],
    [100.0, 200.0, 100.0],
    [256.0, 256.0, 256.0],
], device='cuda', dtype=torch.float32).reshape(1, -1, 3)

print(f"Test queries:")
print(xyz_voxel_test[0])
print()

result = grid_sample_3d(
    mesh.attrs,
    coords_with_batch,
    mesh.voxel_shape,
    xyz_voxel_test,
    mode='trilinear'
)

print(f"Result shape: {result.shape}")
print(f"Result for queries:")
for i in range(4):
    print(f"  Query {i}: {result[0, i].cpu().numpy()}")

print()
print("=" * 70)
print("Testing direct voxel lookup...")

test_coords = mesh.coords[:10]
print(f"First 10 coords (int):")
print(test_coords)

test_query = test_coords.float() + 0.5
test_query = test_query.reshape(1, -1, 3)
print(f"First 10 queries (voxel center):")
print(test_query[0, :5])

result_direct = grid_sample_3d(
    mesh.attrs[:100],
    coords_with_batch[:100],
    mesh.voxel_shape,
    test_query[:5],
    mode='trilinear'
)
print(f"Result from subset (first 5):")
print(result_direct[0, :5])

print()
print("=" * 70)
print("Testing with different coordinate conventions...")

test_pt = mesh.coords[0].float()
print(f"Testing with coord[0] = {test_pt.cpu().numpy()}")

for offset in [0.0, 0.5, 1.0]:
    q = (test_pt + offset).reshape(1, 1, 3)
    r = grid_sample_3d(
        mesh.attrs,
        coords_with_batch,
        mesh.voxel_shape,
        q,
        mode='trilinear'
    )
    print(f"  Query {test_pt.cpu().numpy()} + {offset}: result={r[0, 0].cpu().numpy()}, attrs[0]={mesh.attrs[0].cpu().numpy()}")

print()
print("=" * 70)
print("Checking coordinate indexing order...")
print(f"voxel_shape = {mesh.voxel_shape}")
print(f"Shape format: (batch, channels, W, H, D) or (batch, channels, X, Y, Z)?")
print()

coords_min = mesh.coords.min(dim=0).values
coords_max = mesh.coords.max(dim=0).values
print(f"coords X range: {coords_min[0].item()} - {coords_max[0].item()}")
print(f"coords Y range: {coords_min[1].item()} - {coords_max[1].item()}")
print(f"coords Z range: {coords_min[2].item()} - {coords_max[2].item()}")
print()

print(f"voxel_shape[2] (W/X?): {mesh.voxel_shape[2]}")
print(f"voxel_shape[3] (H/Y?): {mesh.voxel_shape[3]}")
print(f"voxel_shape[4] (D/Z?): {mesh.voxel_shape[4]}")