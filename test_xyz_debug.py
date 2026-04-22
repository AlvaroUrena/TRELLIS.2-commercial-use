#!/usr/bin/env python3
"""
Debug PBR rendering to check xyz_voxel values and grid_sample_3d results.
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
from flex_gemm.ops.grid_sample import grid_sample_3d

os.makedirs(os.environ['TRELLIS_DEBUG_DIR'], exist_ok=True)

print("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Running pipeline...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh.simplify(16777216)

print(f"mesh.coords shape: {mesh.coords.shape}")
print(f"mesh.attrs shape: {mesh.attrs.shape}")
print(f"voxel_shape: {mesh.voxel_shape}")

coords_with_batch = torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1)
print(f"coords_with_batch shape: {coords_with_batch.shape}")

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

renderer = PbrMeshRenderer(
    rendering_options={
        'resolution': 512,
        'near': 1,
        'far': 100,
        'ssaa': 1,
        'peel_layers': 8,
    },
    device='cuda'
)

print("Rendering...")
result = renderer.render(mesh, extrinsics[0], intrinsics[0], envmap)

print(f"result.shaded shape: {result.shaded.shape}")
print(f"result.base_color min/max: {result.base_color.min():.4f} / {result.base_color.max():.4f}")
print(f"result.base_color mean: {result.base_color.mean():.6f}")

nonzero_mask = result.base_color.sum(dim=-1) > 0
print(f"Non-zero base_color pixels: {nonzero_mask.sum().item()} / {nonzero_mask.numel()}")

print()
print("=" * 70)
print("Analyzing debug files...")
print()

debug_dir = os.environ['TRELLIS_DEBUG_DIR']
debug_files = sorted([f for f in os.listdir(debug_dir) if f.endswith('.npy')])
print(f"Found {len(debug_files)} debug files")

xyz_file = None
for f in debug_files:
    if 'DRTK_interp' in f:
        xyz_file = f
        break

if xyz_file:
    xyz = np.load(os.path.join(debug_dir, xyz_file))
    print(f"Loaded {xyz_file}")
    print(f"  Shape: {xyz.shape}")
    print(f"  Range X: {xyz[:, 0].min():.2f} - {xyz[:, 0].max():.2f}")
    print(f"  Range Y: {xyz[:, 1].min():.2f} - {xyz[:, 1].max():.2f}")
    print(f"  Range Z: {xyz[:, 2].min():.2f} - {xyz[:, 2].max():.2f}")
    
    xyz_voxel = (xyz - mesh.origin.cpu().numpy()) / mesh.voxel_size
    print(f"xyz_voxel range:")
    print(f"  X: {xyz_voxel[:, 0].min():.2f} - {xyz_voxel[:, 0].max():.2f}")
    print(f"  Y: {xyz_voxel[:, 1].min():.2f} - {xyz_voxel[:, 1].max():.2f}")
    print(f"  Z: {xyz_voxel[:, 2].min():.2f} - {xyz_voxel[:, 2].max():.2f}")
    
    mesh_coords_min = mesh.coords.min(dim=0).values.cpu().numpy()
    mesh_coords_max = mesh.coords.max(dim=0).values.cpu().numpy()
    print(f"mesh.coords range:")
    print(f"  X: {mesh_coords_min[0]} - {mesh_coords_max[0]}")
    print(f"  Y: {mesh_coords_min[1]} - {mesh_coords_max[1]}")
    print(f"  Z: {mesh_coords_min[2]} - {mesh_coords_max[2]}")
    
    in_range_mask = (
        (xyz_voxel[:, 0] >= mesh_coords_min[0]) & (xyz_voxel[:, 0] <= mesh_coords_max[0]) &
        (xyz_voxel[:, 1] >= mesh_coords_min[1]) & (xyz_voxel[:, 1] <= mesh_coords_max[1]) &
        (xyz_voxel[:, 2] >= mesh_coords_min[2]) & (xyz_voxel[:, 2] <= mesh_coords_max[2])
    )
    print(f"xyl_voxel pixels in mesh.coords range: {in_range_mask.sum()} / {len(in_range_mask)}")
else:
    print("No DRTK_interp file found!")

print()
print("=" * 70)
print("Direct grid_sample_3d test with mesh data...")

xyz_file = os.path.join(debug_dir, '04_DRTK_interp_final_result.npy')
if os.path.exists(xyz_file):
    xyz = np.load(xyz_file)
    xyz_tensor = torch.from_numpy(xyz).cuda().float()
    
    valid_mask = (xyz_tensor[:, 0] != 0) | (xyz_tensor[:, 1] != 0) | (xyz_tensor[:, 2] != 0)
    xyz_valid = xyz_tensor[valid_mask]
    print(f"Non-zero xyz pixels: {len(xyz_valid)}")
    
    if len(xyz_valid) > 100:
        origin = mesh.origin.to(xyz_tensor.device)
        voxel_size = mesh.voxel_size
        
        print(f"origin: {origin}")
        print(f"voxel_size: {voxel_size}")
        
        xyz_voxel = ((xyz_valid[:100] - origin) / voxel_size).reshape(1, -1, 3)
        print(f"First 100 xyz_voxel samples:")
        print(f"  min: {xyz_voxel[0].min(dim=0).values.cpu().numpy()}")
        print(f"  max: {xyz_voxel[0].max(dim=0).values.cpu().numpy()}")
        
        result = grid_sample_3d(
            mesh.attrs,
            coords_with_batch,
            mesh.voxel_shape,
            xyz_voxel,
            mode='trilinear'
        )
        print(f"grid_sample_3d result shape: {result.shape}")
        print(f"grid_sample_3d result min: {result.min():.6f}, max: {result.max():.6f}, mean: {result.mean():.6f}")
        
        nonzero_results = (result.abs() > 1e-6).sum().item()
        print(f"Non-zero samples: {nonzero_results} / {result.numel()}")