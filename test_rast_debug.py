#!/usr/bin/env python3
"""
Debug the rendering pipeline to find where xyz becomes all zeros.
"""

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TRELLIS_DEBUG'] = '1'
os.environ['TRELLIS_DEBUG_DIR'] = 'test_output_pbr/debug'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer, EnvMap
from trellis2.utils.drtk_compat import DRTKContext, interpolate
from trellis2.utils.debug_utils import reset_debug_step, is_debug_enabled, dbg_tensor, dbg_value, next_step

os.makedirs(os.environ['TRELLIS_DEBUG_DIR'], exist_ok=True)

print("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Running pipeline...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh.simplify(16777216)

print(f"mesh.vertices shape: {mesh.vertices.shape}")
print(f"mesh.vertices sample (first 5):")
print(mesh.vertices[:5])

extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

envmap_data = cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
if envmap_data.shape[-1] == 4:
    envmap_data = envmap_data[..., :3]
if envmap_data.dtype == np.uint16:
    envmap_data = envmap_data.astype(np.float32) / 65535.0
elif envmap_data.dtype == np.uint8:
    envmap_data = envmap_data.astype(np.float32) / 255.0
else:
    envmap_data = envmap_data.astype(np.float32)
if envmap_data.ndim == 2:
    envmap_data = np.stack([envmap_data] * 3, axis=-1)
envmap = EnvMap(torch.from_numpy(envmap_data).cuda())

print()
print("=" * 70)
print("Testing DRTK rasterization directly...")
print()

resolution = 512
ctx = DRTKContext()
ctx.set_resolution(resolution)

vertices = mesh.vertices.cuda()
faces = mesh.faces.cuda()
extr = extrinsics[0].cuda()
intr = intrinsics[0].cuda()

print(f"vertices shape: {vertices.shape}, dtype: {vertices.dtype}")
print(f"faces shape: {faces.shape}, dtype: {faces.dtype}")
print(f"extr shape: {extr.shape}")
print(f"intr shape: {intr.shape}")

rast, rast_db = ctx.rasterize(vertices, faces, extr, intr, resolution, near=1.0, far=100.0)

print(f"rast shape: {rast.shape}")
print(f"rast u (barycentric u): min={rast[0, ..., 0].min():.4f}, max={rast[0, ..., 0].max():.4f}, mean={rast[0, ..., 0].mean():.4f}")
print(f"rast v (barycentric v): min={rast[0, ..., 1].min():.4f}, max={rast[0, ..., 1].max():.4f}, mean={rast[0, ..., 1].mean():.4f}")
print(f"rast depth: min={rast[0, ..., 2].min():.4f}, max={rast[0, ..., 2].max():.4f}, mean={rast[0, ..., 2].mean():.4f}")
print(f"rast tri_id: min={rast[0, ..., 3].min():.4f}, max={rast[0, ..., 3].max():.4f}, mean={rast[0, ..., 3].mean():.4f}")

valid_mask = rast[0, ..., 3] > 0
print(f"Valid pixels (tri_id > 0): {valid_mask.sum().item()} / {valid_mask.numel()}")

print()
print("Testing interpolate...")
vertices_orig = mesh.vertices.cuda()
xyz, _ = interpolate(vertices_orig, rast, faces, ctx=ctx)

print(f"xyz shape: {xyz.shape}")
print(f"xyz min: {xyz.min():.6f}, max: {xyz.max():.6f}, mean: {xyz.mean():.6f}")
print(f"xyz valid (non-zero) pixels: {(xyz.abs() > 1e-6).sum().item()} / {xyz.numel()}")

if valid_mask.sum() > 0:
    valid_xyz = xyz[valid_mask.unsqueeze(-1).expand_as(xyz)].reshape(-1, 3)
    print(f"xyz for valid pixels:")
    print(f"  X range: {valid_xyz[:, 0].min():.6f} - {valid_xyz[:, 0].max():.6f}")
    print(f"  Y range: {valid_xyz[:, 1].min():.6f} - {valid_xyz[:, 1].max():.6f}")
    print(f"  Z range: {valid_xyz[:, 2].min():.6f} - {valid_xyz[:, 2].max():.6f}")

print()
print("Testing xyz_voxel computation...")
origin = mesh.origin.cuda()
voxel_size = mesh.voxel_size

print(f"origin: {origin}")
print(f"voxel_size: {voxel_size}")

if valid_mask.sum() > 0:
    valid_xyz_for_voxel = valid_xyz[:10]
    xyz_voxel = ((valid_xyz_for_voxel - origin) / voxel_size)
    print(f"First 10 xyz_voxel values:")
    print(xyz_voxel)
    
    coords_with_batch = torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1)
    print(f"mesh.coords first 10:")
    print(mesh.coords[:10])