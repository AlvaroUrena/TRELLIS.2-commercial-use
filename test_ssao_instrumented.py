#!/usr/bin/env python3
"""
Test SSAO depth values - are they camera-space z or NDC z/w?
This is critical for understanding the 15% darkness difference.
"""
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['TRELLIS_DEBUG'] = '0'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import cv2
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import MeshWithVoxel
from trellis2.utils import render_utils
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer, EnvMap

IMAGE_PATH = "assets/example_image/T.png"
HDRI_PATH = "assets/hdri/forest.exr"

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open(IMAGE_PATH)
mesh_with_voxel = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh_with_voxel.simplify(16777216)

print(f"Generated mesh: {mesh_with_voxel.vertices.shape[0]} vertices, {mesh_with_voxel.faces.shape[0]} faces")

extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
    yaws=[(-16 / 180 * np.pi)],
    pitchs=[20 / 180 * np.pi],
    rs=[10],
    fovs=[8],
)

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
envmap = EnvMap(torch.from_numpy(envmap_data).cuda())

# Create a custom render that captures SSAO inputs
import torch.nn.functional as F
from trellis2.utils.drtk_compat import DepthPeeler
from trellis2.renderers.pbr_mesh_renderer import screen_space_ambient_occlusion, intrinsics_to_projection

# Store original render
original_render = PbrMeshRenderer.render

depth_captured = [None]
normal_captured = [None]
ssao_result = [None]
shaded_before_ssao = [None]

def instrumented_render(self, mesh, extrinsics, intrinsics, envmap, use_envmap_bg=False, transformation=None):
    result = original_render(self, mesh, extrinsics, intrinsics, envmap, use_envmap_bg, transformation)
    # The result.clay is (1 - f_occ) where f_occ is SSAO occlusion
    return result

PbrMeshRenderer.render = instrumented_render

renderer = PbrMeshRenderer(
    rendering_options={
        'resolution': 512,
        'near': 1,
        'far': 100,
        'ssaa': 1,
        'peel_layers': 8,
    },
    device='cuda'
)

result = renderer.render(mesh_with_voxel, extrinsics[0], intrinsics[0], envmap)

# The result.clay is (1 - f_occ)
# f_occ is SSAO occlusion factor
ssao_occlusion = 1 - result.clay

print("\n" + "="*70)
print("SSAO OUTPUT ANALYSIS")
print("="*70)

# Get mask for geometry pixels
mask = result.mask.cpu().numpy()
ssao_occlusion_np = ssao_occlusion.cpu().numpy()

ssao_occlusion_masked = ssao_occlusion_np[mask > 0]

print(f"SSAO occlusion (f_occ):")
print(f"  Geometry pixels: {mask.sum()}")
print(f"  Min: {ssao_occlusion_masked.min():.6f}")
print(f"  Max: {ssao_occlusion_masked.max():.6f}")
print(f"  Mean: {ssao_occlusion_masked.mean():.6f}")
print(f"  Std: {ssao_occlusion_masked.std():.6f}")

# The "clay" output is (1 - f_occ), so pixels brightened by SSAO
# A darker final image means MORE occlusion (higher f_occ)
# If fork SSAO is producing more occlusion, f_occ mean will be higher

print()
print("What would make fork darker:")
print("  If SSAO occlusion is too strong (f_occ too high)")
print("  This could happen if:")
print("    1. Depth values differ between fork and official")
print("    2. Normal values differ")
print("    3. SSAO algorithm differs")
print("    4. View-space reconstruction differs")
print()

# Let's look at shaded vs clay to understand the SSAO effect
shaded_np = result.shaded.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
clay_np = result.clay.cpu().numpy()  # [H, W]

shaded_lum = 0.299 * shaded_np[..., 0] + 0.587 * shaded_np[..., 1] + 0.114 * shaded_np[..., 2]
shaded_lum_masked = shaded_lum[mask > 0]

print(f"Shaded luminance (after SSAO):")
print(f"  Mean: {shaded_lum_masked.mean():.6f}")
print()

# Check official shaded for comparison
official_shaded_path = "test_output_pbr_official/shaded_00.png"
if os.path.exists(official_shaded_path):
    official_shaded = np.array(Image.open(official_shaded_path)).astype(np.float32) / 255.0
    official_lum = 0.299 * official_shaded[..., 0] + 0.587 * official_shaded[..., 1] + 0.114 * official_shaded[..., 2]
    official_lum_masked = official_lum[mask > 0]
    
    print(f"Official shaded luminance:")
    print(f"  Mean: {official_lum_masked.mean():.6f}")
    print()
    
    # If we remove SSAO from both, they should match (if materials/lighting match)
    # shaded_final = shaded_before_ssao * (1 - f_occ)
    # So shaded_before_ssao = shaded_final / (1 - f_occ)
    
    # But we don't have shaded_before_ssao from official...
    # Let's compute what the shaded would be without SSAO
    fork_shaded_before_ssao = shaded_lum / clay_np
    fork_shaded_before_ssao_masked = fork_shaded_before_ssao[mask > 0]
    
    print(f"Fork shaded BEFORE SSAO (estimated):")
    print(f"  Mean: {fork_shaded_before_ssao_masked.mean():.6f}")
    
    # Load official clay
    official_clay_path = "test_output_pbr_official/clay_00.png"
    if os.path.exists(official_clay_path):
        official_clay = np.array(Image.open(official_clay_path)).astype(np.float32) / 255.0
        if official_clay.ndim == 3:
            official_clay = official_clay[..., 0]
        
        official_ssao_occlusion = 1 - official_clay
        official_ssao_occlusion_masked = official_ssao_occlusion[mask > 0]
        
        print()
        print(f"Official SSAO occlusion (f_occ):")
        print(f"  Min: {official_ssao_occlusion_masked.min():.6f}")
        print(f"  Max: {official_ssao_occlusion_masked.max():.6f}")
        print(f"  Mean: {official_ssao_occlusion_masked.mean():.6f}")
        print()
        
        # KEY COMPARISON:
        # If fork f_occ > official f_occ, fork will be darker
        occlusion_ratio = ssao_occlusion_masked.mean() / max(official_ssao_occlusion_masked.mean(), 1e-6)
        print(f"SSAO occlusion ratio (fork/official): {occlusion_ratio:.4f}")
        
        if occlusion_ratio > 1.05:
            print("  -> FORK HAS MORE OCCLUSION - SSAO is causing darkness!")
        elif occlusion_ratio < 0.95:
            print("  -> FORK HAS LESS OCCLUSION - Darkness is NOT from SSAO")
        else:
            print("  -> SSAO occlusion matches - Darkness is elsewhere")
        
        # Compute what shaded would be without SSAO for official
        official_shaded_before_ssao = official_lum / official_clay
        official_shaded_before_ssao_masked = official_shaded_before_ssao[mask > 0]
        
        print()
        print(f"Comparing shaded BEFORE SSAO (should match if materials/lighting match):")
        print(f"  Fork mean: {fork_shaded_before_ssao_masked.mean():.6f}")
        print(f"  Official mean: {official_shaded_before_ssao_masked.mean():.6f}")
        print(f"  Ratio: {fork_shaded_before_ssao_masked.mean() / max(official_shaded_before_ssao_masked.mean(), 1e-6):.4f}")

print()
print("Done.")