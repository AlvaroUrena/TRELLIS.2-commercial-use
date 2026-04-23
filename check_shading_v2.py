#!/usr/bin/env python3
"""
Check clay and shaded values more carefully.
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

fork_shaded = load_img(f"{FORK_DIR}/shaded_00.png")
fork_clay = load_img(f"{FORK_DIR}/clay_00.png")
fork_mask = load_img(f"{FORK_DIR}/mask_pbr_00.png")

official_shaded = load_img(f"{OFFICIAL_DIR}/shaded_00.png")
official_clay = load_img(f"{OFFICIAL_DIR}/clay_00.png")
official_mask = load_img(f"{OFFICIAL_DIR}/mask_pbr_00.png")

fork_geom = fork_mask.max(axis=-1) > 0 if fork_mask.ndim == 3 else fork_mask > 0
official_geom = official_mask.max(axis=-1) > 0 if official_mask.ndim == 3 else official_mask > 0

fork_clay_2d = fork_clay[..., 0] if fork_clay.ndim == 3 else fork_clay
official_clay_2d = official_clay[..., 0] if official_clay.ndim == 3 else official_clay

# Compute f_occ
fork_f_occ = 1 - fork_clay_2d
official_f_occ = 1 - official_clay_2d

# Luminance
def lum(img):
    if img.ndim == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img

fork_lum = lum(fork_shaded)
official_lum = lum(official_shaded)

# Compute pre-SSAO more carefully (avoiding division issues)
# shaded_final = pre_ssao * clay
# Filter by mask and clay > 0.01 to avoid division issues

fork_valid = fork_geom & (fork_clay_2d > 0.01)
official_valid = official_geom & (official_clay_2d > 0.01)

fork_f_occ_v = fork_f_occ[fork_valid]
official_f_occ_v = official_f_occ[official_valid]

fork_clay_v = fork_clay_2d[fork_valid]
official_clay_v = official_clay_2d[official_valid]

fork_lum_v = fork_lum[fork_valid]
official_lum_v = official_lum[official_valid]

fork_pre = fork_lum_v / fork_clay_v
official_pre = official_lum_v / official_clay_v

print("SSAO occlusion (f_occ):")
print(f"  Fork:     mean={fork_f_occ_v.mean():.6f}")
print(f"  Official: mean={official_f_occ_v.mean():.6f}")
print(f"  Ratio: {fork_f_occ_v.mean() / official_f_occ_v.mean():.4f}")
print()

print("Clay (1 - f_occ):")
print(f"  Fork:     mean={fork_clay_v.mean():.6f}")
print(f"  Official: mean={official_clay_v.mean():.6f}")
print(f"  Ratio: {fork_clay_v.mean() / official_clay_v.mean():.4f}")
print()

print("Final luminance:")
print(f"  Fork:     mean={fork_lum_v.mean():.6f}")
print(f"  Official: mean={official_lum_v.mean():.6f}")
print(f"  Ratio: {fork_lum_v.mean() / official_lum_v.mean():.4f}")
print()

print("Pre-SSAO (shaded / clay):")
print(f"  Fork:     mean={fork_pre.mean():.6f}, std={fork_pre.std():.6f}")
print(f"  Official: mean={official_pre.mean():.6f}, std={official_pre.std():.6f}")
print(f"  Ratio: {fork_pre.mean() / official_pre.mean():.4f}")
print()

# The ratio of brightness should be pre_ssao * clay
expected = fork_pre.mean() / official_pre.mean() * (fork_clay_v.mean() / official_clay_v.mean())
actual = fork_lum_v.mean() / official_lum_v.mean()
print(f"Expected ratio: {expected:.4f}")
print(f"Actual ratio:   {actual:.4f}")