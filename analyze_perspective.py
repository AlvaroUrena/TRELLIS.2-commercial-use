#!/usr/bin/env python3
"""
Compare perspective matrices and depth values between fork and official.
"""
import torch
import numpy as np
from PIL import Image
import os

print("=" * 70)
print("PERSPECTIVE MATRIX AND DEPTH ANALYSIS")
print("=" * 70)

# The perspective matrix is used by SSAO to reconstruct view positions
# fx = perspective[0, 0], fy = perspective[1, 1], cx = perspective[0, 2], cy = perspective[1, 2]

# From parameters: yaws=[-16°], pitchs=[20°], rs=[10], fovs=[8]
# r = 10 means distance from camera is 10 (camera-space Z)
# fov = 8 degrees

# Let's compute what the perspective matrix should look like
fov = 8  # degrees
near = 1
far = 100
resolution = 512

# Intrinsics-to-projection is standard:
# For fov_y based projection:
# fx = fy = resolution / (2 * tan(fov/2))
# But with fov=8° and typical OpenGL projection:

fov_rad = np.radians(fov)
f = 1.0 / np.tan(fov_rad / 2)

print(f"\nCamera parameters:")
print(f"  FOV: {fov}°")
print(f"  Near: {near}")
print(f"  Far: {far}")
print(f"  Resolution: {resolution}")
print(f"  tan(fov/2) = {np.tan(fov_rad / 2):.6f}")
print(f"  f = 1/tan(fov/2) = {f:.6f}")

# Standard OpenGL projection (fov_y based)
# For resolution=512, aspect=1, fov=8°:
# perspective[0,0] = f
# perspective[1,1] = f
# perspective[2,2] = (far+near)/(far-near)
# perspective[2,3] = 2*far*near/(far-near)
# perspective[3,2] = 1

print(f"\nExpected perspective matrix:")
print(f"  fx = {f:.6f}")
print(f"  fy = {f:.6f}")
print(f"  cx = 0 (centered)")
print(f"  cy = 0 (centered)")

# In SSAO:
# x_view = (x_grid - cx) * depth / fx
# y_view = (y_grid - cy) * depth / fy
# view_pos = [x_view, y_view, depth]

print(f"\nSSAO depth reconstruction:")
print(f"  x_grid ranges from -1 to 1 (NDC)")
print(f"  x_view = x_grid * depth / fx")
print(f"  With depth ~10 and fx ~{f:.2f}:")
print(f"  x_view ~ x_grid * {10/f:.2f}")

# Now check what the actual depth values are
# Depth values should be camera-space z from vertices_camera[..., 2]
# For r=10, yaws=[-16°], pitchs=[20°]:
# The distance from camera should be around 10 (since r is the distance)

print(f"\nExpected camera-space Z: ~{10} (from r parameter)")

# Let's check the clay images to understand SSAO output
FORK_CLAY = "test_output_pbr/clay_00.png"
OFFICIAL_CLAY = "test_output_pbr_official/clay_00.png"

if os.path.exists(FORK_CLAY) and os.path.exists(OFFICIAL_CLAY):
    fork_clay = np.array(Image.open(FORK_CLAY)).astype(np.float32) / 255.0
    official_clay = np.array(Image.open(OFFICIAL_CLAY)).astype(np.float32) / 255.0
    
    fork_f_occ = 1 - (fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay)
    official_f_occ = 1 - (official_clay[..., 0] if official_clay.ndim == 3 else official_clay)
    
    fork_mask = np.array(Image.open("test_output_pbr/mask_pbr_00.png")) > 0
    official_mask = np.array(Image.open("test_output_pbr_official/mask_pbr_00.png")) > 0
    
    fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
    official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0
    
    fork_f_occ_geom = fork_f_occ[fork_geom]
    official_f_occ_geom = official_f_occ[official_geom]
    
    print(f"\nSSAO occlusion values:")
    print(f"  Fork f_occ:     mean={fork_f_occ_geom.mean():.4f}, std={fork_f_occ_geom.std():.4f}")
    print(f"  Official f_occ: mean={official_f_occ_geom.mean():.4f}, std={official_f_occ_geom.std():.4f}")
    print(f"  Ratio (fork/official): {fork_f_occ_geom.mean() / official_f_occ_geom.mean():.4f}")
    
    # Compute SSAO intensity adjustment needed
    # SSAO applies: shaded *= (1 - f_occ)
    # f_occ = occlusion_count / samples * intensity
    # To match official: fork_intensity = official_intensity * (official_f_occ / fork_f_occ)
    print(f"\nSSAO intensity adjustment:")
    print(f"  Current intensity: 1.5")
    print(f"  To match official occlusion, fork intensity should be:")
    adjusted_intensity = 1.5 * (official_f_occ_geom.mean() / fork_f_occ_geom.mean())
    print(f"  1.5 * (0.232 / 0.261) = {adjusted_intensity:.4f}")

print()
print("=" * 70)
print("KEY INSIGHT:")
print("Both implementations use vertices_camera[..., 2:3] for SSAO depth.")
print("This should be identical camera-space z values (~10).")
print("But SSAO produces 12.4% more occlusion in fork.")
print("=" * 70)
print()
print("POSSIBLE CAUSES:")
print("1. Perspective matrix differences in how cx/cy are applied")
print("2. Grid normalization differences in SSAO screen-space sampling")
print("3. Normal vector differences affecting sample directions")
print()
print("FIX OPTIONS:")
print("1. Adjust SSAO intensity from 1.5 to 1.33 (quickest fix)")
print("2. Debug the exact depth values passed to SSAO")
print("3. Verify normal vectors match exactly")