#!/usr/bin/env python3
"""
Test PBR shading with SSAO disabled to isolate shading issues.
"""
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['TRELLIS_DEBUG'] = '1'
os.environ['TRELLIS_DEBUG_DIR'] = 'test_output_pbr_nossao/debug'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs('test_output_pbr_nossao', exist_ok=True)
os.makedirs(os.environ['TRELLIS_DEBUG_DIR'], exist_ok=True)

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
OUTPUT_DIR = "test_output_pbr_nossao"

# Modify the SSAO function in-place
import trellis2.renderers.pbr_mesh_renderer as pbr_module

# Save original
original_render = pbr_module.PbrMeshRenderer.render

def render_with_ssao_disabled(self, mesh, extrinsics, intrinsics, envmap, use_envmap_bg=False, transformation=None):
    """Render with SSAO disabled by patching during this call."""
    # Call original but save SSAO result and restore after
    result = original_render(self, mesh, extrinsics, intrinsics, envmap, use_envmap_bg, transformation)
    # The SSAO is already applied, so we can't easily undo it here
    return result

# Just modify the SSAO function globally
pbr_module.screen_space_ambient_occlusion = lambda *args, **kwargs: args[0] * 0  # Return zeros

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

def save_image(tensor, name):
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
        arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    Image.fromarray(arr).save(os.path.join(OUTPUT_DIR, f"{name}_00.png"))

save_image(result.shaded, "shaded")
save_image(result.base_color, "base_color")
save_image(result.metallic, "metallic")
save_image(result.roughness, "roughness")
save_image(result.alpha, "alpha")
save_image(result.normal, "normal_pbr")
save_image(result.mask, "mask_pbr")

print(f"Shaded (no SSAO): min={result.shaded.min().item():.4f}, max={result.shaded.max().item():.4f}, mean={result.shaded.mean().item():.4f}")
print(f"Done! Outputs in {OUTPUT_DIR}/")