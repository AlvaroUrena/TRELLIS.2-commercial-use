#!/usr/bin/env python3
"""
Test to verify SSAO depth scaling issue.
- Fork uses camera-space z (~10)
- Official uses NDC z/w (~0.82)
- SSAO radius needs to be scaled for the depth convention difference.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image

# Compare depth values
fork_mask = np.array(Image.open("test_output_pbr/mask_pbr_00.png")) > 0
official_mask = np.array(Image.open("test_output_pbr_official/mask_pbr_00.png")) > 0
combined_mask = (fork_mask | official_mask)

# Check debug files for depth values
debug_dir = "test_output_pbr/debug"
debug_official = "test_output_pbr_official/debug"

print("SSAO Depth Analysis")
print("=" * 60)

# Look for depth-related files
import os as os_module
fork_files = sorted(os_module.listdir(debug_dir)) if os_module.path.exists(debug_dir) else []
official_files = sorted(os_module.listdir(debug_official)) if os_module.path.exists(debug_official) else []

print(f"Fork debug files (first 20): {fork_files[:20]}")
print(f"Official debug files (first 20): {official_files[:20]}")
print()

# Load depth from both
fork_depth_paths = [f for f in fork_files if 'depth' in f.lower()]
official_depth_paths = [f for f in official_files if 'depth' in f.lower()]

print(f"Fork depth files: {fork_depth_paths}")
print(f"Official depth files: {official_depth_paths}")

# The key insight: DRTK depth is camera-space, nvdiffrast depth is NDC
# Camera-space z for this scene: ~9.66 to 10.27 (mean ~10)
# NDC z/w for this scene: ~0.81 to 0.82 (mean ~0.815)

# SSAO radius of 0.1:
# - In fork: samples within 0.1/10 = 1% of scene depth
# - In official: samples within 0.1/0.82 = 12% of scene depth
# 
# To match official: fork should use radius = 0.1 * (10/0.82) ≈ 1.22

print()
print("Depth scaling factor:")
print(f"  Fork depth range:     ~[9.66, 10.27] (camera-space)")
print(f"  Official depth range:  ~[0.81, 0.82] (NDC z/w)")
print(f"  Typical ratio:        {10.0/0.82:.2f}")
print()
print(f"Current SSAO radius:   0.1")
print(f"Recommended fork radius: {0.1 * 10.0/0.82:.3f}")
print()
print("FIX: Scale SSAO radius by DEPTH_SCALE = camera_z / ndc_z ≈ 12.2")