#!/usr/bin/env python3
"""
Compare PBR shading intermediate values.
Both pre-SSAO shading and SSAO contribute to the darkness.
"""
import torch
import numpy as np
from PIL import Image
import os

FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_img(path):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path))
    return arr.astype(np.float32) / 255.0

print("=" * 70)
print("INVESTIGATING PRE-SSAO SHADING DARKNESS")
print("=" * 70)
print()

# The pre-SSAO shading is ~10% darker in fork
# This could be due to:
# 1. Diffuse irradiance differences
# 2. Specular prefilter differences
# 3. FG LUT differences
# 4. Normal differences affecting dot products

fork_mask = load_img(f"{FORK_DIR}/mask_pbr_00.png")
official_mask = load_img(f"{OFFICIAL_DIR}/mask_pbr_00.png")

fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0

# Check if normals match
fork_normal = load_img(f"{FORK_DIR}/normal_pbr_00.png")
official_normal = load_img(f"{OFFICIAL_DIR}/normal_pbr_00.png")

if fork_normal is not None and official_normal is not None:
    fork_normal_geom = fork_normal[fork_geom]  # [N_fork, 3]
    official_normal_geom = official_normal[official_geom]  # [N_official, 3]
    
    # Use minimum count for comparison
    n_compare = min(len(fork_normal_geom), len(official_normal_geom))
    
    print("Normal comparison (camera space):")
    print(f"  Fork geometry pixels: {len(fork_normal_geom)}")
    print(f"  Official geometry pixels: {len(official_normal_geom)}")
    print(f"  Comparing {n_compare} pixels")
    
    # Check if normals match
    diff_normal = np.abs(fork_normal_geom[:n_compare] - official_normal_geom[:n_compare])
    
    print(f"  Fork:     min={fork_normal_geom[:n_compare].min():.4f}, max={fork_normal_geom[:n_compare].max():.4f}, mean={fork_normal_geom[:n_compare].mean():.4f}")
    print(f"  Official: min={official_normal_geom[:n_compare].min():.4f}, max={official_normal_geom[:n_compare].max():.4f}, mean={official_normal_geom[:n_compare].mean():.4f}")
    print(f"  Diff:     mean={diff_normal.mean():.6f}, max={diff_normal.max():.6f}")
    
    # For camera-space normals, a difference in Y coordinate would indicate Y-flip issues
    print(f"  Per-channel diff: R={diff_normal[..., 0].mean():.6f}, G={diff_normal[..., 1].mean():.6f}, B={diff_normal[..., 2].mean():.6f}")
    
    if diff_normal.mean() > 0.01:
        print("  -> NORMALS DIFFER! This affects SSAO sampling directions.")
    else:
        print("  -> Normals match within tolerance.")
    print()

# Compare RGB channels of shaded output to see which channels are darker
fork_shaded = load_img(f"{FORK_DIR}/shaded_00.png")
official_shaded = load_img(f"{OFFICIAL_DIR}/shaded_00.png")

if fork_shaded is not None and official_shaded is not None:
    fork_shaded_geom = fork_shaded[fork_geom]
    official_shaded_geom = official_shaded[official_geom]
    
    n_compare_shaded = min(len(fork_shaded_geom), len(official_shaded_geom))
    
    print("Shaded RGB comparison:")
    print(f"  Fork RGB:     R={fork_shaded_geom[:n_compare_shaded, 0].mean():.4f}, G={fork_shaded_geom[:n_compare_shaded, 1].mean():.4f}, B={fork_shaded_geom[:n_compare_shaded, 2].mean():.4f}")
    print(f"  Official RGB: R={official_shaded_geom[:n_compare_shaded, 0].mean():.4f}, G={official_shaded_geom[:n_compare_shaded, 1].mean():.4f}, B={official_shaded_geom[:n_compare_shaded, 2].mean():.4f}")
    print(f"  Ratio per channel: R={fork_shaded_geom[:n_compare_shaded, 0].mean()/official_shaded_geom[:n_compare_shaded, 0].mean():.4f}, G={fork_shaded_geom[:n_compare_shaded, 1].mean()/official_shaded_geom[:n_compare_shaded, 1].mean():.4f}, B={fork_shaded_geom[:n_compare_shaded, 2].mean()/official_shaded_geom[:n_compare_shaded, 2].mean():.4f}")
    print()

# Compare clay (SSAO survival factor)
fork_clay = load_img(f"{FORK_DIR}/clay_00.png")
official_clay = load_img(f"{OFFICIAL_DIR}/clay_00.png")

if fork_clay is not None and official_clay is not None:
    fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
    official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay
    
    fork_clay_geom = fork_clay_2d[fork_geom]
    official_clay_geom = official_clay_2d[official_geom]
    
    print("Clay (1 - SSAO occlusion) comparison:")
    print(f"  Fork:     min={fork_clay_geom.min():.4f}, max={fork_clay_geom.max():.4f}, mean={fork_clay_geom.mean():.4f}")
    print(f"  Official: min={official_clay_geom.min():.4f}, max={official_clay_geom.max():.4f}, mean={official_clay_geom.mean():.4f}")
    print(f"  Ratio: {fork_clay_geom.mean() / official_clay_geom.mean():.4f}")
    print(f"  (Fork has {(official_clay_geom.mean() - fork_clay_geom.mean()):.4f} less survival factor)")
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("Two issues found:")
print("1. Pre-SSAO shading is ~10% darker in fork")
print("   -> Likely issue with PBREnvironmentLight.shade() or cube_to_dir()")
print()
print("2. SSAO occlusion is ~12% higher in fork")  
print("   -> Likely issue with SSAO sampling (different normals or seed differences)")
print()
print("NEXT STEPS:")
print("- Compare PBREnvironmentLight.shade() outputs")
print("- Compare diffuse irradiance and specular prefilter")
print("- Verify cube_to_dir() mapping consistency between implementations")