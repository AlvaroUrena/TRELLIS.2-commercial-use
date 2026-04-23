import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['TRELLIS_DEBUG'] = '1'
os.environ['TRELLIS_DEBUG_DIR'] = 'test_output_pbr/debug'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import torch
from PIL import Image
import cv2
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import MeshWithVoxel
from trellis2.utils import render_utils
from trellis2.utils.debug_utils import reset_debug_step, is_debug_enabled
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer, EnvMap

IMAGE_PATH = "assets/example_image/T.png"
HDRI_PATH = "assets/hdri/forest.exr"
OUTPUT_DIR = "test_output_pbr"
SEED = 42
PIPELINE_TYPE = "512"
RENDER_RESOLUTION = 512
N_VIEWS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ['TRELLIS_DEBUG_DIR'], exist_ok=True)

reset_debug_step()

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open(IMAGE_PATH)
mesh_with_voxel = pipeline.run(image, seed=SEED, pipeline_type=PIPELINE_TYPE)[0]
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
        'resolution': RENDER_RESOLUTION,
        'near': 1,
        'far': 100,
        'ssaa': 1,
        'peel_layers': 8,
    },
    device='cuda'
)

for i, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
    result = renderer.render(mesh_with_voxel, extr, intr, envmap)
    
    def save_image(tensor, name, is_hdr=False):
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            arr = tensor.detach().cpu().numpy()
        if is_hdr:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        Image.fromarray(arr).save(os.path.join(OUTPUT_DIR, f"{name}_{i:02d}.png"))
    
    save_image(result.shaded, "shaded", is_hdr=True)
    save_image(result.normal, "normal_pbr")
    save_image(result.base_color, "base_color")
    save_image(result.metallic, "metallic")
    save_image(result.roughness, "roughness")
    save_image(result.alpha, "alpha")
    save_image(result.mask, "mask_pbr")
    if hasattr(result, 'clay'):
        save_image(result.clay, "clay")
    
    print(f"  View {i}: PBR rendered")
    print(f"    shaded: {result.shaded.min().item():.4f} - {result.shaded.max().item():.4f}")
    print(f"    base_color: {result.base_color.min().item():.4f} - {result.base_color.max().item():.4f}")
    print(f"    metallic: {result.metallic.min().item():.4f} - {result.metallic.max().item():.4f}")
    print(f"    roughness: {result.roughness.min().item():.4f} - {result.roughness.max().item():.4f}")
    print(f"    alpha: {result.alpha.min().item():.4f} - {result.alpha.max().item():.4f}")
    print(f"    mask coverage: {(result.mask > 0).sum().item()} / {result.mask.numel()}")

print(f"\nDone! PBR outputs in {OUTPUT_DIR}/")
print(f"Debug .npy files saved in {os.environ['TRELLIS_DEBUG_DIR']}/")