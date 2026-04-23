#!/usr/bin/env python3
"""
Compare SSAO occlusion between fork and official outputs.
"""
import numpy as np
from PIL import Image
import os

FORK_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_img(path):
    """Load image as float32 in [0, 1] range."""
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path))
    return arr.astype(np.float32) / 255.0

def compute_luminance(img):
    """Compute luminance from RGB image."""
    if img.ndim == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img

print("=" * 70)
print("COMPARING SSAO OCCLUSION")
print("=" * 70)
print()

# Load images
fork_shaded = load_img(f"{FORK_DIR}/shaded_00.png")
fork_mask = load_img(f"{FORK_DIR}/mask_pbr_00.png")
fork_clay = load_img(f"{FORK_DIR}/clay_00.png") if os.path.exists(f"{FORK_DIR}/clay_00.png") else None

official_shaded = load_img(f"{OFFICIAL_DIR}/shaded_00.png")
official_mask = load_img(f"{OFFICIAL_DIR}/mask_pbr_00.png")
official_clay = load_img(f"{OFFICIAL_DIR}/clay_00.png") if os.path.exists(f"{OFFICIAL_DIR}/clay_00.png") else None

if fork_mask is None or official_mask is None:
    print("ERROR: Could not load masks")
    exit(1)

# Get geometry mask
fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0
combined_geom = fork_geom | official_geom

print(f"Geometry pixels: {combined_geom.sum()} / {combined_geom.size}")
print()

# Compare shaded
fork_lum = compute_luminance(fork_shaded)
official_lum = compute_luminance(official_shaded)

fork_lum_geom = fork_lum[fork_geom]
official_lum_geom = official_lum[official_geom]

print("Shaded luminance comparison:")
print(f"  Fork (geometry):     mean={fork_lum_geom.mean():.6f}, std={fork_lum_geom.std():.6f}")
print(f"  Official (geometry): mean={official_lum_geom.mean():.6f}, std={official_lum_geom.std():.6f}")
print(f"  Ratio (fork/official): {fork_lum_geom.mean() / official_lum_geom.mean():.4f}")
print()

# Compare SSAO "clay" output (which is 1 - f_occ)
if fork_clay is not None and official_clay is not None:
    fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
    official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay
    
    fork_f_occ = 1 - fork_clay_2d  # SSAO occlusion factor
    official_f_occ = 1 - official_clay_2d
    
    fork_f_occ_geom = fork_f_occ[fork_geom]
    official_f_occ_geom = official_f_occ[official_geom]
    
    print("SSAO occlusion comparison:")
    print(f"  Fork (f_occ):     mean={fork_f_occ_geom.mean():.6f}, std={fork_f_occ_geom.std():.6f}")
    print(f"  Official (f_occ): mean={official_f_occ_geom.mean():.6f}, std={official_f_occ_geom.std():.6f}")
    print(f"  Ratio (fork/official): {fork_f_occ_geom.mean() / official_f_occ_geom.mean():.4f}")
    print()
    
    # SSAO makes output darker by multiplying by (1 - f_occ)
    # If fork f_occ > official f_occ, fork will be darker
    
    # Check if darkness ratio matches SSAO ratio
    # shaded_final = shaded_before_ssao * (1 - f_occ)
    # fork_shaded = fork_before_ssao * fork_clay
    # official_shaded = official_before_ssao * official_clay
    
    # Assuming shaded_before_ssao is the same for both (materials match)
    # brightness_ratio = fork_clay / official_clay
    
    clay_ratio = (fork_clay_2d[fork_geom]).mean() / (official_clay_2d[official_geom]).mean()
    print(f"Clay ratio (fork/official): {clay_ratio:.4f}")
    print()
    
    # Estimate shaded before SSAO (avoiding division by zero)
    eps = 1e-6
    fork_clay_safe = np.maximum(fork_clay_2d, eps)
    official_clay_safe = np.maximum(official_clay_2d, eps)
    
    fork_before_ssao = fork_lum / fork_clay_safe
    official_before_ssao = official_lum / official_clay_safe
    
    fork_before_ssao_geom = fork_before_ssao[fork_geom]
    official_before_ssao_geom = official_before_ssao[official_geom]
    
    # Filter out NaN/Inf
    fork_before_ssao_geom = fork_before_ssao_geom[np.isfinite(fork_before_ssao_geom)]
    official_before_ssao_geom = official_before_ssao_geom[np.isfinite(official_before_ssao_geom)]
    
    print("Estimated shaded BEFORE SSAO:")
    print(f"  Fork:     mean={fork_before_ssao_geom.mean():.6f}, std={fork_before_ssao_geom.std():.6f}")
    print(f"  Official: mean={official_before_ssao_geom.mean():.6f}, std={official_before_ssao_geom.std():.6f}")
    print(f"  Ratio (fork/official): {fork_before_ssao_geom.mean() / official_before_ssao_geom.mean():.4f}")
    print()
    
    print("=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    
    if abs(fork_f_occ_geom.mean() - official_f_occ_geom.mean()) < 0.01:
        print("SSAO occlusion values MATCH (within 1%)")
        print("-> Darkness is NOT from SSAO")
        print("-> Investigate: diffuse irradiance, specular prefilter, or FG LUT")
    elif fork_f_occ_geom.mean() > official_f_occ_geom.mean():
        print("Fork has MORE SSAO occlusion than official")
        print(f"  -> This causes darker output: factor = {(1 - fork_f_occ_geom.mean()) / (1 - official_f_occ_geom.mean()):.4f}")
        print("  -> Root cause: SSAO depth convention or parameters need adjustment")
    else:
        print("Fork has LESS SSAO occlusion than official")
        print("  -> Darkness is NOT from SSAO")
        print("  -> Investigate elsewhere")

elif fork_clay is None:
    print("Fork clay_00.png not found - need to run PBR render with clay output")
elif official_clay is None:
    print("Official clay_00.png not found - need to run PBR render with clay output")

print()
print("Done.")