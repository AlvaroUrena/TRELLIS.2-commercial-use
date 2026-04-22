#!/usr/bin/env python3
"""
Trace grid_sample_3d values through the PBR render pipeline.
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
from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
from flex_gemm.ops.grid_sample import grid_sample_3d

print("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Running pipeline...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh.simplify(16777216)

# Setup
extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

resolution = 512
perspective = intrinsics_to_projection(intrinsics[0], 1.0, 100.0)
full_proj = (perspective @ extrinsics[0]).unsqueeze(0)
extr = extrinsics[0].unsqueeze(0)

vertices = mesh.vertices.unsqueeze(0).cuda()
vertices_orig = vertices.clone()
vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
faces = mesh.faces.cuda()

coords_with_batch = torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1).cuda()

print(f"mesh.origin: {mesh.origin}")
print(f"mesh.voxel_size: {mesh.voxel_size}")
print(f"mesh.voxel_shape: {mesh.voxel_shape}")
print(f"mesh.layout: {mesh.layout}")
print(f"mesh.attrs mean: {mesh.attrs.mean():.6f}")
print(f"mesh.attrs min: {mesh.attrs.min():.6f}")
print(f"mesh.attrs max: {mesh.attrs.max():.6f}")
print()

class MockGLctx:
    pass

with DepthPeeler(MockGLctx(), vertices_clip, faces, (resolution, resolution)) as peeler:
    rast, _ = peeler.rasterize_next_layer()
    
    mask = rast[..., -1:] > 0
    valid_mask = mask[0, ..., 0]
    print(f"mask shape: {mask.shape}")
    print(f"Valid pixels: {valid_mask.sum().item()}")
    print()
    
    # Interpolate vertices
    xyz = interpolate(vertices_orig, rast, faces, peeler=peeler)[0]
    print(f"xyz shape: {xyz.shape}")
    print(f"xyz at valid pixels: min={xyz[0, valid_mask].min():.4f}, max={xyz[0, valid_mask].max():.4f}")
    
    # Convert to voxel coords
    origin = mesh.origin.cuda()
    voxel_size = mesh.voxel_size
    xyz_voxel = ((xyz - origin) / voxel_size)
    
    print(f"xyz_voxel shape: {xyz_voxel.shape}")
    xyz_voxel_valid = xyz_voxel[0, valid_mask]
    print(f"xyz_voxel at valid: X=[{xyz_voxel_valid[:, 0].min():.2f}, {xyz_voxel_valid[:, 0].max():.2f}]")
    print(f"                  Y=[{xyz_voxel_valid[:, 1].min():.2f}, {xyz_voxel_valid[:, 1].max():.2f}]")
    print(f"                  Z=[{xyz_voxel_valid[:, 2].min():.2f}, {xyz_voxel_valid[:, 2].max():.2f}]")
    print()
    
    # Reshape for grid_sample_3d
    xyz_voxel_flat = xyz_voxel.reshape(1, -1, 3)
    print(f"xyz_voxel_flat shape: {xyz_voxel_flat.shape}")
    
    # Call grid_sample_3d
    img = grid_sample_3d(
        mesh.attrs.cuda(),
        coords_with_batch,
        mesh.voxel_shape,
        xyz_voxel_flat,
        mode='trilinear'
    )
    
    print(f"img shape after grid_sample_3d: {img.shape}")
    print(f"img at all pixels: min={img.min():.6f}, max={img.max():.6f}, mean={img.mean():.6f}")
    
    # Reshape
    img_reshaped = img.reshape(1, resolution, resolution, mesh.attrs.shape[-1])
    print(f"img_reshaped shape: {img_reshaped.shape}")
    print(f"img_reshaped at valid pixels: min={img_reshaped[0, valid_mask].min():.6f}, max={img_reshaped[0, valid_mask].max():.6f}, mean={img_reshaped[0, valid_mask].mean():.6f}")
    
    # Apply mask
    img_masked = img_reshaped * mask
    print(f"img_masked shape: {img_masked.shape}")
    print(f"img_masked at all pixels: min={img_masked.min():.6f}, max={img_masked.max():.6f}, mean={img_masked.mean():.6f}")
    print(f"img_masked at valid pixels: min={img_masked[0, valid_mask].min():.6f}, max={img_masked[0, valid_mask].max():.6f}, mean={img_masked[0, valid_mask].mean():.6f}")
    
    # Extract base_color channel
    base_color_channels = img_masked[0, ..., mesh.layout['base_color']]
    print()
    print(f"base_color_channels shape: {base_color_channels.shape}")
    print(f"base_color at valid pixels:")
    bc_valid = base_color_channels[valid_mask]
    print(f"  R: min={bc_valid[:, 0].min():.4f}, max={bc_valid[:, 0].max():.4f}, mean={bc_valid[:, 0].mean():.4f}")
    print(f"  G: min={bc_valid[:, 1].min():.4f}, max={bc_valid[:, 1].max():.4f}, mean={bc_valid[:, 1].mean():.4f}")
    print(f"  B: min={bc_valid[:, 2].min():.4f}, max={bc_valid[:, 2].max():.4f}, mean={bc_valid[:, 2].mean():.4f}")