#!/usr/bin/env python3
"""
Complete test of grid_sample_3d pipeline with actual mesh and rendering.
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
import cv2
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.utils.drtk_compat import DepthPeeler, interpolate
from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection, EnvMap
from flex_gemm.ops.grid_sample import grid_sample_3d

print("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Running pipeline...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh.simplify(16777216)

print(f"mesh.vertices: {mesh.vertices.shape}")
print(f"mesh.faces: {mesh.faces.shape}")
print(f"mesh.coords: {mesh.coords.shape}, range: {mesh.coords.min().item()}-{mesh.coords.max().item()}")
print(f"mesh.attrs: {mesh.attrs.shape}")
print(f"mesh.origin: {mesh.origin}")
print(f"mesh.voxel_size: {mesh.voxel_size}")
print(f"mesh.voxel_shape: {mesh.voxel_shape}")

# Get camera parameters
extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

resolution = 512
near = 1.0
far = 100.0

perspective = intrinsics_to_projection(intrinsics[0], near, far)
full_proj = (perspective @ extrinsics[0]).unsqueeze(0)
extr = extrinsics[0].unsqueeze(0)

# Transform vertices
vertices = mesh.vertices.unsqueeze(0).cuda()
vertices_orig = vertices.clone()
vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
vertices_camera = torch.bmm(vertices_homo, extr.transpose(-1, -2))
vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
faces = mesh.faces.cuda()

# Create DepthPeeler and render
class MockGLctx:
    pass

glctx = MockGLctx()

print()
print("Running DepthPeeler rasterization...")

with DepthPeeler(glctx, vertices_clip, faces, (resolution, resolution)) as peeler:
    rast, rast_db = peeler.rasterize_next_layer()
    
    print(f"rast shape: {rast.shape}")
    print(f"  tri_id > 0 pixels: {(rast[0, ..., 3] > 0).sum().item()}")
    
    # Interpolate vertices to get xyz
    xyz = interpolate(vertices_orig, rast, faces, peeler=peeler)[0]
    print(f"xyz shape: {xyz.shape}")
    
    # Check valid pixels
    mask = rast[..., -1:] > 0
    valid_mask = mask[0, ..., 0]
    
    xyz_valid = xyz[0, valid_mask]
    print(f"Valid xyz pixels: {xyz_valid.shape[0]}")
    print(f"xyz X range: {xyz_valid[:, 0].min():.4f} - {xyz_valid[:, 0].max():.4f}")
    print(f"xyz Y range: {xyz_valid[:, 1].min():.4f} - {xyz_valid[:, 1].max():.4f}")
    print(f"xyz Z range: {xyz_valid[:, 2].min():.4f} - {xyz_valid[:, 2].max():.4f}")
    
    # Convert to voxel coordinates
    origin = mesh.origin.to(xyz.device)
    voxel_size = mesh.voxel_size
    
    print()
    print(f"origin: {origin}")
    print(f"voxel_size: {voxel_size}")
    
    xyz_voxel = ((xyz - origin) / voxel_size)
    print(f"xyz_voxel shape: {xyz_voxel.shape}")
    
    xyz_voxel_valid = xyz_voxel[0, valid_mask]
    print(f"xyz_voxel X range: {xyz_voxel_valid[:, 0].min():.2f} - {xyz_voxel_valid[:, 0].max():.2f}")
    print(f"xyz_voxel Y range: {xyz_voxel_valid[:, 1].min():.2f} - {xyz_voxel_valid[:, 1].max():.2f}")
    print(f"xyz_voxel Z range: {xyz_voxel_valid[:, 2].min():.2f} - {xyz_voxel_valid[:, 2].max():.2f}")
    
    # Compare with mesh.coords range
    coords = mesh.coords.cuda()
    print()
    print(f"mesh.coords X range: {coords[:, 0].min().item()} - {coords[:, 0].max().item()}")
    print(f"mesh.coords Y range: {coords[:, 1].min().item()} - {coords[:, 1].max().item()}")
    print(f"mesh.coords Z range: {coords[:, 2].min().item()} - {coords[:, 2].max().item()}")
    
    # Test grid_sample_3d with a small subset
    print()
    print("=" * 70)
    print("Testing grid_sample_3d...")
    
    xyz_voxel_flat = xyz_voxel.reshape(1, -1, 3)
    coords_with_batch = torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1).cuda()
    
    print(f"xyz_voxel_flat shape: {xyz_voxel_flat.shape}")
    print(f"coords_with_batch shape: {coords_with_batch.shape}")
    print(f"mesh.attrs shape: {mesh.attrs.shape}")
    print(f"mesh.voxel_shape: {mesh.voxel_shape}")
    
    # Sample at mesh.coords centers (these should definitely return non-zero)
    test_coords = mesh.coords[:10].float().cuda() + 0.5
    test_query = test_coords.reshape(1, -1, 3)
    print()
    print("Testing direct coord queries (coord + 0.5):")
    result_test = grid_sample_3d(mesh.attrs.cuda(), coords_with_batch, mesh.voxel_shape, test_query, mode='trilinear')
    print(f"  result shape: {result_test.shape}")
    print(f"  result[0, 0]: {result_test[0, 0].cpu().numpy()}")
    print(f"  attrs[0]: {mesh.attrs[0].cpu().numpy()}")
    
    # Now test with actual xyz_voxel values
    print()
    print("Testing with actual xyz_voxel values...")
    
    # Take first 100 valid xyz_voxel values
    xyz_voxel_100 = xyz_voxel_flat[:, :100, :]
    print(f"First 100 xyz_voxel samples:")
    print(f"  sample 0: {xyz_voxel_100[0, 0].cpu().numpy()}")
    print(f"  sample 1: {xyz_voxel_100[0, 1].cpu().numpy()}")
    print(f"  sample 2: {xyz_voxel_100[0, 2].cpu().numpy()}")
    
    result_100 = grid_sample_3d(mesh.attrs.cuda(), coords_with_batch, mesh.voxel_shape, xyz_voxel_100, mode='trilinear')
    print(f"  result shape: {result_100.shape}")
    print(f"  result mean: {result_100.mean():.6f}")
    print(f"  result min: {result_100.min():.6f}")
    print(f"  result max: {result_100.max():.6f}")
    print(f"  nonzero ratio: {(result_100.abs() > 1e-6).sum().item()} / {result_100.numel()}")
    
    # Full grid sample
    print()
    print("Running full grid_sample_3d...")
    img = grid_sample_3d(
        mesh.attrs.cuda(),
        coords_with_batch,
        mesh.voxel_shape,
        xyz_voxel_flat,
        mode='trilinear'
    )
    
    img_reshaped = img.reshape(1, resolution, resolution, mesh.attrs.shape[-1]) * mask
    
    print(f"img shape: {img.shape}")
    print(f"img mean: {img.mean():.6f}")
    print(f"img min: {img.min():.6f}")
    print(f"img max: {img.max():.6f}")
    
    # Check at valid pixels
    img_valid = img_reshaped[0, valid_mask]
    print(f"img at valid pixels shape: {img_valid.shape}")
    print(f"img at valid pixels mean: {img_valid.mean():.6f}")
    print(f"img at valid pixels min: {img_valid.min():.6f}")
    print(f"img at valid pixels max: {img_valid.max():.6f}")