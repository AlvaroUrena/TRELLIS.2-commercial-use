#!/usr/bin/env python3
"""
Detailed comparison of shading components.
"""
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
print("DETAILED SHADING COMPONENT ANALYSIS")
print("=" * 70)

# Load images
fork_shaded = load_img(f"{FORK_DIR}/shaded_00.png")
fork_clay = load_img(f"{FORK_DIR}/clay_00.png")
fork_mask = load_img(f"{FORK_DIR}/mask_pbr_00.png")

official_shaded = load_img(f"{OFFICIAL_DIR}/shaded_00.png")
official_clay = load_img(f"{OFFICIAL_DIR}/clay_00.png")
official_mask = load_img(f"{OFFICIAL_DIR}/mask_pbr_00.png")

# Get masks
fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0

# Compute occlusion factors
fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay

fork_f_occ = 1 - fork_clay_2d
official_f_occ = 1 - official_clay_2d

# Compute luminance
def luminance(img):
    if img.ndim == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img

fork_shaded_lum = luminance(fork_shaded)
official_shaded_lum = luminance(official_shaded)

# Estimated pre-SSAO shading (shaded / clay)
eps = 1e-8
fork_pre_ssao = fork_shaded_lum / (fork_clay_2d + eps)
official_pre_ssao = official_shaded_lum / (official_clay_2d + eps)

# Masked values
fork_f_occ_geom = fork_f_occ[fork_geom]
official_f_occ_geom = official_f_occ[official_geom]

fork_clay_geom = fork_clay_2d[fork_geom]
official_clay_geom = official_clay_2d[official_geom]

fork_pre_ssao_geom = fork_pre_ssao[fork_geom]
official_pre_ssao_geom = official_pre_ssao[official_geom]

fork_shaded_lum_geom = fork_shaded_lum[fork_geom]
official_shaded_lum_geom = official_shaded_lum[official_geom]

print("\nFINAL SHADED (AFTER SSAO):")
print(f"  Fork luminance:     mean={fork_shaded_lum_geom.mean():.6f}")
print(f"  Official luminance: mean={official_shaded_lum_geom.mean():.6f}")
print(f"  Ratio: {fork_shaded_lum_geom.mean() / official_shaded_lum_geom.mean():.4f}")

print("\nSSAO OCCLUSION FACTOR (f_occ):")
print(f"  Fork f_occ:     mean={fork_f_occ_geom.mean():.6f}")
print(f"  Official f_occ: mean={official_f_occ_geom.mean():.6f}")
print(f"  Ratio: {fork_f_occ_geom.mean() / official_f_occ_geom.mean():.4f}")
print(f"  (Fork has {100 * (fork_f_occ_geom.mean() / official_f_occ_geom.mean() - 1):.1f}% MORE occlusion)")

print("\nSSAO SURVIVAL FACTOR (clay = 1 - f_occ):")
print(f"  Fork clay:     mean={fork_clay_geom.mean():.6f}")
print(f"  Official clay: mean={official_clay_geom.mean():.6f}")
print(f"  Ratio: {fork_clay_geom.mean() / official_clay_geom.mean():.4f}")
print(f"  (Fork clay is {100 * (1 - fork_clay_geom.mean() / official_clay_geom.mean()):.1f}% SMALLER)")

print("\nESTIMATED PRE-SSAO SHADING:")
print(f"  Fork pre-SSAO:     mean={fork_pre_ssao_geom[np.isfinite(fork_pre_ssao_geom)].mean():.6f}")
print(f"  Official pre-SSAO: mean={official_pre_ssao_geom[np.isfinite(official_pre_ssao_geom)].mean():.6f}")
pre_ssao_ratio = fork_pre_ssao_geom[np.isfinite(fork_pre_ssao_geom)].mean() / official_pre_ssao_geom[np.isfinite(official_pre_ssao_geom)].mean()
print(f"  Ratio: {pre_ssao_ratio:.4f}")
print(f"  (Fork is {100 * (1 - pre_ssao_ratio):.1f}% DARKER before SSAO)")

print("\n" + "=" * 70)
print("FORMULA:")
print("  shaded_final = pre_ssao * clay")
print("  shaded_final = pre_ssao * (1 - f_occ)")
print("=" * 70)
print()

expected_ratio = pre_ssao_ratio * (fork_clay_geom.mean() / official_clay_geom.mean())
print(f"Expected shaded ratio: {pre_ssao_ratio:.4f} * {fork_clay_geom.mean() / official_clay_geom.mean():.4f} = {expected_ratio:.4f}")
print(f"Observed shaded ratio: {fork_shaded_lum_geom.mean() / official_shaded_lum_geom.mean():.4f}")
print()

if abs(pre_ssao_ratio - 1.0) > 0.05:
    print("ISSUE 1: Pre-SSAO shading differs by {:.1f}%".format(abs(1 - pre_ssao_ratio) * 100))
    print("  -> Need to investigate: diffuse irradiance, specular prefilter, FG LUT")

if abs(fork_f_occ_geom.mean() / official_f_occ_geom.mean() - 1.0) > 0.05:
    print("ISSUE 2: SSAO occlusion differs by {:.1f}%".format(abs(1 - fork_f_occ_geom.mean() / official_f_occ_geom.mean()) * 100))
    print("  -> Need to investigate: SSAO depth scaling or intensity")

print()
print("CONCLUSION:")
pre_ssao_diff = abs(1 - pre_ssao_ratio) * 100
ssao_diff = (fork_f_occ_geom.mean() / official_f_occ_geom.mean() - 1) * 100
if abs(pre_ssao_ratio - 1.0) > 0.05 and abs(fork_f_occ_geom.mean() / official_f_occ_geom.mean() - 1.0) > 0.05:
    print("BOTH issues contribute to darkness:")
    print(f"  {pre_ssao_diff:.1f}% from pre-SSAO shading being darker")
    print(f"  {ssao_diff:.1f}% from SSAO being more aggressive")
elif abs(pre_ssao_ratio - 1.0) > 0.05:
    print("Pre-SSAO shading is the main issue")
else:
    print("SSAO is the main issue")