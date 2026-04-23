#!/usr/bin/env python3
"""
Compare SSAO intermediate values: depth input and occlusion output.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image

FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_npy_or_none(path):
    if os.path.exists(path):
        return np.load(path)
    return None

def load_png(path):
    if os.path.exists(path):
        return np.array(Image.open(path)).astype(np.float32) / 255.0
    return None

print("SSAO Depth and Occlusion Comparison")
print("=" * 70)

# Load mask for filtering
fork_mask = load_png(f"{FORK_DIR}/mask_pbr_00.png")
official_mask = load_png(f"{OFFICIAL_DIR}/mask_pbr_00.png")

if fork_mask is None or official_mask is None:
    print("ERROR: Could not load masks")
    exit(1)

combined_mask = ((fork_mask > 0) | (official_mask > 0)).flatten()
print(f"Combined mask: {combined_mask.sum()} / {combined_mask.size} pixels")
print()

# Check shaded images to understand SSAO effect
fork_shaded = load_png(f"{FORK_DIR}/shaded_00.png")
official_shaded = load_png(f"{OFFICIAL_DIR}/shaded_00.png")

if fork_shaded is not None and official_shaded is not None:
    print("Shaded (final) comparison:")
    
    # Compute luminance
    fork_lum = 0.299 * fork_shaded[..., 0] + 0.587 * fork_shaded[..., 1] + 0.114 * fork_shaded[..., 2]
    official_lum = 0.299 * official_shaded[..., 0] + 0.587 * official_shaded[..., 1] + 0.114 * official_shaded[..., 2]
    
    fork_geom_mask = fork_mask > 0
    official_geom_mask = official_mask > 0
    fork_lum_masked = fork_lum[fork_geom_mask[..., 0] if fork_geom_mask.ndim == 3 else fork_geom_mask]
    official_lum_masked = official_lum[official_geom_mask[..., 0] if official_geom_mask.ndim == 3 else official_geom_mask]
    
    print(f"  Fork luminance: min={fork_lum_masked.min():.4f}, max={fork_lum_masked.max():.4f}, mean={fork_lum_masked.mean():.4f}")
    print(f"  Official luminance: min={official_lum_masked.min():.4f}, max={official_lum_masked.max():.4f}, mean={official_lum_masked.mean():.4f}")
    print(f"  Ratio (fork/official): {fork_lum_masked.mean() / max(official_lum_masked.mean(), 1e-6):.4f}")
    print()

# The key question: what DEPTH value is passed to SSAO?
# Both implementations use vertices_camera[..., 2:3] which should be camera-space z
# Let's check by looking at the perspective matrix and how depth is reconstructed

# Looking at the SSAO function:
# x_view = (x_grid - cx) * depth / fx
# y_view = (y_grid - cy) * depth / fy  
# view_pos = [x_view, y_view, depth]

# This reconstructs view-space position assuming depth is the z-coordinate in view space.
# If depth is camera-space z (distance from camera), this works.
# If depth were NDC z/w, this would NOT work correctly.

# Both implementations SHOULD use camera-space z from vertices_camera interpolation.

# Let me check the depth files in debug
fork_depth_files = [f for f in os.listdir(f"{FORK_DIR}/debug") if 'depth' in f.lower()] if os.path.exists(f"{FORK_DIR}/debug") else []
official_depth_files = [f for f in os.listdir(f"{OFFICIAL_DIR}/debug") if 'depth' in f.lower()] if os.path.exists(f"{OFFICIAL_DIR}/debug") else []

print(f"Fork depth debug files: {fork_depth_files}")
print(f"Official depth debug files: {official_depth_files}")
print()

# The main difference might not be in SSAO depth scaling but elsewhere
# Let me check if there's a clay output (which is just SSAO occlusion)
fork_clay = load_png(f"{FORK_DIR}/clay_00.png") if os.path.exists(f"{FORK_DIR}/clay_00.png") else None
official_clay = load_png(f"{OFFICIAL_DIR}/clay_00.png") if os.path.exists(f"{OFFICIAL_DIR}/clay_00.png") else None

if fork_clay is not None and official_clay is not None:
    print("Clay (SSAO occlusion) comparison:")
    
    fork_clay_masked = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
    official_clay_masked = official_clay[..., 0] if official_clay.ndim == 3 else official_clay
    
    fork_clay_flat = fork_clay_masked.flatten()
    official_clay_flat = official_clay_masked.flatten()
    
    print(f"  Fork clay: min={fork_clay_masked[combined_mask.reshape(fork_clay_masked.shape) > 0].min():.4f}, max={fork_clay_masked[combined_mask.reshape(fork_clay_masked.shape) > 0].max():.4f}, mean={fork_clay_masked[combined_mask.reshape(fork_clay_masked.shape) > 0].mean():.4f}")
    print(f"  Official clay: min={official_clay_masked[combined_mask.reshape(official_clay_masked.shape) > 0].min():.4f}, max={official_clay_masked[combined_mask.reshape(official_clay_masked.shape) > 0].max():.4f}, mean={official_clay_masked[combined_mask.reshape(official_clay_masked.shape) > 0].mean():.4f}")
    
    # Clay should show (1 - f_occ), so higher values = less occlusion
    # If fork clay is lower, it means MORE occlusion (darker SSAO)
    print()

# Key insight: Both should use camera-space z for SSAO depth
# Let me check if there's any depth-related debug output

# Check if there's interpolated depth saved
fork_npz_depth_files = [f for f in os.listdir(f"{FORK_DIR}/debug") if 'gb_depth' in f.lower()] if os.path.exists(f"{FORK_DIR}/debug") else []
official_npz_depth_files = [f for f in os.listdir(f"{OFFICIAL_DIR}/debug") if 'gb_depth' in f.lower()] if os.path.exists(f"{OFFICIAL_DIR}/debug") else []

print(f"Fork gb_depth files: {fork_npz_depth_files}")
print(f"Official gb_depth files: {official_npz_depth_files}")

if fork_npz_depth_files and official_npz_depth_files:
    fork_depth = np.load(f"{FORK_DIR}/debug/{fork_npz_depth_files[0]}")
    official_depth = np.load(f"{OFFICIAL_DIR}/debug/{official_npz_depth_files[0]}")
    
    print("\nDepth values from gb_depth:")
    print(f"  Fork: min={fork_depth.min():.4f}, max={fork_depth.max():.4f}, mean={fork_depth.mean():.4f}")
    print(f"  Official: min={official_depth.min():.4f}, max={official_depth.max():.4f}, mean={official_depth.mean():.4f}")
    
    if abs(fork_depth.mean() - official_depth.mean()) < 0.1:
        print("  -> Depth values MATCH! SSAO should behave the same.")
    else:
        print("  -> Depth values DIFFER! This is the issue.")
        print(f"  -> Ratio: {fork_depth.mean() / max(official_depth.mean(), 1e-6):.2f}")

print()
print("=" * 70)
print("CONCLUSION:")
print("Both implementations should use camera-space z for SSAO depth.")
print("If depth values differ, there's a bug in the depth interpolation.")
print("If depth values match, SSAO should produce identical occlusion.")