#!/usr/bin/env python3
"""
Test that compares SSAO output between fork and official by modifying PbrMeshRenderer.
"""
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs('test_output_ssao', exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import cv2
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import MeshWithVoxel
from trellis2.utils import render_utils
from trellis2.utils.debug_utils import reset_debug_step
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer, EnvMap, screen_space_ambient_occlusion

IMAGE_PATH = "assets/example_image/T.png"
HDRI_PATH = "assets/hdri/forest.exr"
OUTPUT_DIR = "test_output_ssao"

reset_debug_step()
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open(IMAGE_PATH)
mesh_with_voxel = pipeline.run(image, seed=42, pipeline_type="512")[0]
mesh_with_voxel.simplify(16777216)

print(f"Generated mesh: {mesh_with_voxel.vertices.shape[0]} vertices, {mesh_with_voxel.faces.shape[0]} faces")

reset_debug_step()

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

# Store original render function
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer
original_render = PbrMeshRenderer.render

# Create modified render that saves SSAO and depth
def debug_render(self, mesh, extrinsics, intrinsics, envmap, use_envmap_bg=False, transformation=None):
    # We need to access internal variables - let's just run once
    result = original_render(self, mesh, extrinsics, intrinsics, envmap, use_envmap_bg, transformation)
    return result

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

# Save outputs
from trellis2.utils.drtk_compat import RasterizeCudaContext, DepthPeeler
from trellis2.utils import drtk_compat
from trellis2.representations.mesh import MeshWithVoxel
import torch.nn.functional as F
from easydict import EasyDict as edict

# Let's reconstruct what SSAO sees by manually invoking it with test data
# We need depth and normal in the format SSAO expects

# The result includes normal, which we can use
# Depth is from vertices_camera[..., 2:3]

print("\nTesting SSAO parameters:")
print(f"  depth range should be ~[9.66, 10.27] for DRTK camera-space z")
print(f"  SSAO radius=0.1, intensity=1.5")

# Create a simple test depth to understand SSAO scaling
test_depth = torch.ones(512, 512, 1, device='cuda') * 10.0  # Typical depth
test_normal = torch.zeros(512, 512, 3, device='cuda')
test_normal[..., 2] = 1.0  # Forward facing normals

from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
perspective = intrinsics_to_projection(intrinsics[0], 1, 100)
print(f"  perspective matrix:\n{perspective}")

# Apply SSAO with test data
test_ssao = screen_space_ambient_occlusion(test_depth, test_normal, perspective, radius=0.1, intensity=1.5)
print(f"  Test SSAO (uniform depth 10): min={test_ssao.min():.4f}, max={test_ssao.max():.4f}, mean={test_ssao.mean():.4f}")

# Compare with what official SSAO would produce with NDC depth ~0.82
test_depth_ndc = torch.ones(512, 512, 1, device='cuda') * 0.82
test_ssao_ndc = screen_space_ambient_occlusion(test_depth_ndc, test_normal, perspective, radius=0.1, intensity=1.5)
print(f"  Test SSAO (NDC depth 0.82): min={test_ssao_ndc.min():.4f}, max={test_ssao_ndc.max():.4f}, mean={test_ssao_ndc.mean():.4f}")

print("\nDone!")