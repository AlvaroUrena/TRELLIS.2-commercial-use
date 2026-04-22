#!/usr/bin/env python3
"""
Debug DRTK rasterization to find where xyz becomes zeros.
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
from trellis2.utils.drtk_compat import DRTKContext, interpolate
import drtk

print("Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Running pipeline...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh.simplify(16777216)

vertices = mesh.vertices.cuda()
faces = mesh.faces.cuda().to(torch.int32)

print(f"vertices shape: {vertices.shape}, dtype: {vertices.dtype}")
print(f"faces shape: {faces.shape}, dtype: {faces.dtype}")
print(f"vertices range: min={vertices.min():.4f}, max={vertices.max():.4f}")

extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

resolution = 512
extr = extrinsics[0].cuda()
intr = intrinsics[0].cuda()

near = 1.0
far = 100.0

perspective = torch.zeros((4, 4), device='cuda', dtype=torch.float32)
perspective[0, 0] = intr[0, 0] * 2 / resolution
perspective[1, 1] = intr[1, 1] * 2 / resolution
perspective[2, 0] = (resolution - 2 * intr[0, 2]) / resolution
perspective[2, 1] = (2 * intr[1, 2] - resolution) / resolution
perspective[2, 2] = -(far + near) / (far - near)
perspective[2, 3] = -1.0
perspective[3, 2] = -2 * far * near / (far - near)

full_proj = perspective @ extr
vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
vertices_clip = (vertices_homo @ full_proj.T)

print(f"vertices_clip shape: {vertices_clip.shape}")
print(f"vertices_clip range: min={vertices_clip.min():.4f}, max={vertices_clip.max():.4f}")

w_clip = vertices_clip[..., 3].clamp(min=1e-8, max=1e8)
x_ndc = vertices_clip[..., 0] / w_clip
y_ndc = -vertices_clip[..., 1] / w_clip

h, w = resolution, resolution
x_pix = (x_ndc + 1) * 0.5 * w - 0.5
y_pix = (h - 1) - ((y_ndc + 1) * 0.5 * h - 0.5)
z_cam = vertices_clip[..., 3].clone()

v_pix = torch.stack([x_pix, y_pix, z_cam], dim=-1)
v_pix = v_pix.unsqueeze(0)

print(f"v_pix shape: {v_pix.shape}")
print(f"v_pix range: x={v_pix[0, :, 0].min():.2f}-{v_pix[0, :, 0].max():.2f}, y={v_pix[0, :, 1].min():.2f}-{v_pix[0, :, 1].max():.2f}")

print()
print("Calling DRTK rasterize...")
index_img = drtk.rasterize(v_pix, faces, height=h, width=w)
print(f"index_img shape: {index_img.shape}, dtype: {index_img.dtype}")

index_img_2d = index_img[0] if index_img.dim() == 3 else index_img
valid_pixels = (index_img_2d >= 0).sum().item()
print(f"Valid pixels (index_img >= 0): {valid_pixels} / {h * w}")
print(f"index_img unique values: {torch.unique(index_img_2d).shape}")
print(f"index_img valid range: min={index_img_2d[index_img_2d >= 0].min()}, max={index_img_2d.max()}")

print()
print("Calling DRTK render...")
depth, bary = drtk.render(v_pix, faces, index_img)
print(f"depth shape: {depth.shape}")
print(f"bary shape: {bary.shape}")

valid_mask = index_img_2d >= 0
print(f"depth at valid pixels: min={depth[0][valid_mask].min():.4f}, max={depth[0][valid_mask].max():.4f}")

print()
print("Building rast tensor...")
batch_size = v_pix.shape[0]
rast = torch.zeros(batch_size, h, w, 4, device='cuda', dtype=torch.float32)
rast[0, ..., 0] = bary[0, 1]
rast[0, ..., 1] = bary[0, 2]
rast[0, ..., 2] = depth[0]
rast[0, ..., 3] = (index_img.float() + 1).float()[0] if index_img.dim() == 3 else (index_img.float() + 1)

print(f"rast shape: {rast.shape}")
print(f"rast[0, ..., 3] unique positive values: {(rast[0, ..., 3] > 0).sum().item()}")

print()
print("Testing interpolate on vertices...")
vertices_input = vertices.unsqueeze(0)

index_img_input = index_img if index_img.dim() == 3 else index_img.unsqueeze(0)
bary_input = bary if bary.dim() == 4 else bary.unsqueeze(0)

result = drtk.interpolate(vertices_input, faces, index_img_input, bary_input)
print(f"Interpolate result shape: {result.shape}")
print(f"Result dtype: {result.dtype}")

result_permuted = result.permute(0, 2, 3, 1)
print(f"Result permuted shape: {result_permuted.shape}")

print()
print("Checking result values at valid pixels...")
valid_idx = index_img_2d[valid_mask][:10].long()
if len(valid_idx) > 0:
    valid_positions = torch.where(valid_mask)
    for i in range(min(10, len(valid_idx))):
        py, px = valid_positions[0][i], valid_positions[1][i]
        tri_id = index_img_2d[py, px]
        bary_weights = bary[0, :, py, px]
        xyz_at_pixel = result_permuted[0, py, px]
        print(f"  Pixel ({py}, {px}): tri_id={tri_id}, bary={bary_weights.cpu().numpy()}, xyz={xyz_at_pixel.cpu().numpy()}")

print()
print("Full result stats:")
print(f"  Result min: {result.min():.6f}")
print(f"  Result max: {result.max():.6f}")
print(f"  Result mean: {result.mean():.6f}")
print(f"  Result abs mean: {result.abs().mean():.6f}")

bg_mask = index_img_2d < 0
result_zeroed = result_permuted.clone()
result_zeroed[0, bg_mask] = 0.0

print(f"After zeroing background:")
print(f"  Result min: {result_zeroed.min():.6f}")
print(f"  Result max: {result_zeroed.max():.6f}")
print(f"  Result mean: {result_zeroed.mean():.6f}")
print(f"  Non-zero entries: {(result_zeroed.abs() > 1e-8).sum().item()} / {result_zeroed.numel()}")