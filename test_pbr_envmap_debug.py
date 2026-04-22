#!/usr/bin/env python3
"""
Debug PBR shading to compare intermediate values between fork and official.
"""
import torch
import numpy as np
from PIL import Image
import cv2
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from trellis2.renderers.pbr_envmap import PBREnvironmentLight, sample_cubemap
from trellis2.renderers.pbr_mesh_renderer import EnvMap

# Load HDRI
HDRI_PATH = "assets/hdri/forest.exr"
envmap_data = cv2.imread(HDRI_PATH, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
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

envmap_tensor = torch.from_numpy(envmap_data).cuda()

# Build PBR envmap
print("Building PBR environment light...")
envmap = EnvMap(envmap_tensor)
pbr_light = envmap._backend

# Test directions (normalized)
test_dirs = torch.tensor([
    [0, 1, 0],      # +Y (up)
    [0, -1, 0],     # -Y (down)
    [1, 0, 0],      # +X (right)
    [-1, 0, 0],     # -X (left)
    [0, 0, 1],      # +Z (forward)
    [0, 0, -1],     # -Z (back)
], dtype=torch.float32, device='cuda')
test_dirs = test_dirs / test_dirs.norm(dim=-1, keepdim=True)

print("\nTesting sample_cubemap:")
for i, d in enumerate(test_dirs):
    result = sample_cubemap(pbr_light.diffuse, d.unsqueeze(0))
    print(f"  Dir {i} {d.tolist()}: diffuse={result.squeeze().tolist()}")

print("\nTesting specular mip levels:")
# Test at different roughness values
roughness_values = [0.1, 0.3, 0.5, 0.7, 1.0]
for rough in roughness_values:
    mip = pbr_light.get_mip(torch.tensor([[rough]], device='cuda'))
    print(f"  Roughness {rough}: mip level = {mip.item():.2f}")

print("\nTesting shade() with single direction:")
# Create a simple test case
gb_pos = torch.zeros(1, 1, 1, 3, device='cuda')
gb_normal = torch.tensor([[[[0, 0, 1]]]], dtype=torch.float32, device='cuda')  # Facing forward
kd = torch.ones(1, 1, 1, 3, device='cuda') * 0.5  # 50% albedo
ks = torch.zeros(1, 1, 1, 3, device='cuda')  # ORM
ks[..., 1] = 0.3  # 30% roughness
ks[..., 2] = 0.8  # 80% metallic
view_pos = torch.tensor([[[[0, 0, -10]]]], dtype=torch.float32, device='cuda')  # Camera in front

result = pbr_light.shade(gb_pos, gb_normal, kd, ks, view_pos)
print(f"  Result: {result.squeeze().tolist()}")

print("\nChecking diffuse cubemap values:")
print(f"  Diffuse shape: {pbr_light.diffuse.shape}")
print(f"  Diffuse min: {pbr_light.diffuse.min():.4f}, max: {pbr_light.diffuse.max():.4f}, mean: {pbr_light.diffuse.mean():.4f}")

print("\nChecking specular cubemap values:")
for i, spec in enumerate(pbr_light.specular):
    print(f"  Specular mip {i}: shape={spec.shape}, min={spec.min():.4f}, max={spec.max():.4f}, mean={spec.mean():.4f}")

print("\nDone.")