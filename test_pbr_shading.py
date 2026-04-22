#!/usr/bin/env python3
"""
Diagnostic test to compare PBR shading intermediate values between fork and official.
"""

import os
import sys
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image

def load_npy(name):
    """Load .npy file from both fork and official directories."""
    fork_path = f"test_output_pbr/debug/{name}"
    official_path = f"test_output_pbr_official/debug/{name}"
    
    fork = np.load(fork_path) if os.path.exists(fork_path) else None
    official = np.load(official_path) if os.path.exists(official_path) else None
    
    return fork, official

def compare_tensor(name, fork, official, mask=None):
    """Compare two tensors and print statistics."""
    if fork is None or official is None:
        print(f"  {name}: MISSING")
        return
    
    if fork.shape != official.shape:
        print(f"  {name}: SHAPE MISMATCH - fork {fork.shape} vs official {official.shape}")
        return
    
    diff = np.abs(fork - official)
    
    if mask is not None:
        diff = diff[mask]
        fork = fork[mask]
        official = official[mask]
    
    print(f"  {name}:")
    print(f"    fork:      min={fork.min():.6f}, max={fork.max():.6f}, mean={fork.mean():.6f}")
    print(f"    official:  min={official.min():.6f}, max={official.max():.6f}, mean={official.mean():.6f}")
    print(f"    diff:      mean={diff.mean():.6f}, max={diff.max():.6f}")
    
    # Check for NaN/Inf
    if np.isnan(fork).any() or np.isnan(official).any():
        print(f"    WARNING: NaN values detected!")
    if np.isinf(fork).any() or np.isinf(official).any():
        print(f"    WARNING: Inf values detected!")

def main():
    print("=" * 70)
    print("PBR SHADING INTERMEDIATE COMPARISON")
    print("=" * 70)
    print()
    
    # Load mask
    fork_mask_path = "test_output_pbr/mask_pbr_00.png"
    official_mask_path = "test_output_pbr_official/mask_pbr_00.png"
    
    fork_mask = np.array(Image.open(fork_mask_path)) > 0
    official_mask = np.array(Image.open(official_mask_path)) > 0
    combined_mask = (fork_mask | official_mask).flatten()
    
    print(f"Combined mask: {combined_mask.sum()} / {combined_mask.size} pixels")
    print()
    
    # Check what debug files are available
    debug_dir = "test_output_pbr/debug"
    if os.path.exists(debug_dir):
        files = sorted(os.listdir(debug_dir))
        print(f"Available debug files in fork: {files[:20]}...")
        
    debug_dir_official = "test_output_pbr_official/debug"
    if os.path.exists(debug_dir_official):
        files_official = sorted(os.listdir(debug_dir_official))
        print(f"Available debug files in official: {files_official[:20]}...")
    print()
    
    # Compare key intermediate values if available
    tensors_to_compare = [
        ('gb_depth.npy', 'Depth'),
        ('gb_normal.npy', 'Normal (world)'),
        ('gb_cam_normal.npy', 'Normal (camera)'),
        ('gb_pos.npy', 'Position'),
        ('gb_basecolor.npy', 'Base color'),
        ('gb_metallic.npy', 'Metallic'),
        ('gb_roughness.npy', 'Roughness'),
        ('diffuse_lookup.npy', 'Diffuse irradiance'),
        ('specular_lookup.npy', 'Specular prefilter'),
        ('fg_lookup.npy', 'FG LUT'),
        ('shaded_before_ssao.npy', 'Shaded (before SSAO)'),
        ('f_occ.npy', 'SSAO occlusion'),
    ]
    
    for filename, name in tensors_to_compare:
        fork, official = load_npy(filename)
        if fork is not None:
            # Flatten spatial dimensions
            if fork.ndim == 4:  # [1, H, W, C]
                fork = fork[0].reshape(-1, fork.shape[-1])
                official = official[0].reshape(-1, official.shape[-1])
            elif fork.ndim == 3:  # [1, H, W] or [H, W, C]
                if fork.shape[0] == 1:
                    fork = fork[0]
                    official = official[0]
                fork = fork.reshape(-1, fork.shape[-1] if fork.ndim > 1 else -1)
                official = official.reshape(-1, official.shape[-1] if official.ndim > 1 else -1)
            
            compare_tensor(name, fork, official, combined_mask)
            print()

if __name__ == "__main__":
    main()