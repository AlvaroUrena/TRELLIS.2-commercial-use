#!/usr/bin/env python3
"""
Compare PBR rendering outputs between fork (DRTK + PyTorch PBR) and reference (nvdiffrast + nvdiffrec).

Usage:
    python compare_pbr.py
    
Compares:
    test_output_pbr/          - Fork output
    test_output_pbr_official/ - Reference output
"""

import numpy as np
from PIL import Image
import os

PBR_DIR = "test_output_pbr"
OFFICIAL_DIR = "test_output_pbr_official"

def load_image(path):
    """Load image as float array in [0, 1] range."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    if arr.dtype == np.uint8:
        arr = arr / 255.0
    elif arr.dtype == np.uint16:
        arr = arr / 65535.0
    return arr

def compare_images(name, fork_path, official_path):
    """Compare two images and print statistics."""
    if not os.path.exists(fork_path):
        print(f"  {name}: FORK FILE NOT FOUND: {fork_path}")
        return None
    if not os.path.exists(official_path):
        print(f"  {name}: OFFICIAL FILE NOT FOUND: {official_path}")
        return None
    
    fork = load_image(fork_path)
    official = load_image(official_path)
    
    if fork.shape != official.shape:
        print(f"  {name}: SHAPE MISMATCH - fork {fork.shape} vs official {official.shape}")
        return None
    
    diff = np.abs(fork - official)
    
    # Filter by mask (only compare where there's geometry)
    fork_mask = fork.sum(axis=-1) > 0 if fork.ndim == 3 else fork > 0
    official_mask = official.sum(axis=-1) > 0 if official.ndim == 3 else official > 0
    combined_mask = fork_mask | official_mask
    
    if combined_mask.sum() == 0:
        combined_mask = np.ones_like(combined_mask)
    
    masked_diff = diff[combined_mask] if diff.ndim == 1 else diff[combined_mask]
    
    print(f"  {name}:")
    print(f"    fork:      min={fork.min():.4f}, max={fork.max():.4f}, mean={fork.mean():.4f}")
    print(f"    official:  min={official.min():.4f}, max={official.max():.4f}, mean={official.mean():.4f}")
    print(f"    diff:      mean={masked_diff.mean():.4f}, max={masked_diff.max():.4f}")
    
    return {
        'fork_min': float(fork.min()),
        'fork_max': float(fork.max()),
        'fork_mean': float(fork.mean()),
        'official_min': float(official.min()),
        'official_max': float(official.max()),
        'official_mean': float(official.mean()),
        'diff_mean': float(masked_diff.mean()),
        'diff_max': float(masked_diff.max()),
    }

def main():
    print("=" * 70)
    print("PBR OUTPUT COMPARISON")
    print("=" * 70)
    print()
    
    results = {}
    
    # Compare rendered outputs
    images = [
        ('shaded_00.png', 'Shaded (final render)'),
        ('base_color_00.png', 'Base color'),
        ('metallic_00.png', 'Metallic'),
        ('roughness_00.png', 'Roughness'),
        ('alpha_00.png', 'Alpha'),
        ('normal_pbr_00.png', 'Normal'),
        ('mask_pbr_00.png', 'Mask'),
    ]
    
    for filename, display_name in images:
        fork_path = os.path.join(PBR_DIR, filename)
        official_path = os.path.join(OFFICIAL_DIR, filename)
        result = compare_images(display_name, fork_path, official_path)
        if result:
            results[filename] = result
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if 'shaded_00.png' in results:
        r = results['shaded_00.png']
        brightness_ratio = r['fork_mean'] / max(r['official_mean'], 1e-6)
        print(f"Shaded brightness ratio (fork/official): {brightness_ratio:.3f}")
        
        if brightness_ratio < 0.8:
            print("  -> FORK IS DIMMER - investigate PBREnvironmentLight.shade()")
        elif brightness_ratio > 1.2:
            print("  -> FORK IS BRIGHTER - investigate PBREnvironmentLight.shade()")
        else:
            print("  -> Brightness matches within tolerance")
    
    if 'base_color_00.png' in results:
        r = results['base_color_00.png']
        if r['fork_max'] < 0.6 * r['official_max']:
            print(f"Base color range mismatch: fork max {r['fork_max']:.3f} vs official max {r['official_max']:.3f}")
            print("  -> Investigate grid_sample_3d or voxel attribute interpolation")
    
    print()
    print("Done.")

if __name__ == "__main__":
    main()