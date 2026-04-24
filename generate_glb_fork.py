#!/usr/bin/env python
"""
Generate GLB from fork (DRTK) implementation with fixed seed.
"""
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

import sys
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import glb_utils
import o_voxel

SEED = 42
IMAGE_PATH = Path("assets/example_image/T.png")
OUTPUT_DIR = Path("glb_comparison")

print(f"[FORK] Generating GLB with seed={SEED}")
print(f"[FORK] Input: {IMAGE_PATH}")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load pipeline
print("[FORK] Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# Load image and run
print("[FORK] Running inference...")
image = Image.open(IMAGE_PATH)
mesh = pipeline.run(image, seed=SEED)[0]
mesh.simplify(16777216)

print(f"[FORK] Mesh generated: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

# Export to GLB
print("[FORK] Exporting GLB...")
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000,
    texture_size=4096,
    remesh=True,
    remesh_band=1,
    remesh_project=0,
    verbose=True
)

glb_path = OUTPUT_DIR / "fork_sample.glb"
glb_utils.export_glb_fixed(glb, str(glb_path), extension_webp=True)

print(f"[FORK] GLB saved to: {glb_path}")
print(f"[FORK] File size: {glb_path.stat().st_size:,} bytes")