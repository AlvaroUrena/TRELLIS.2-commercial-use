#!/usr/bin/env python
"""
Compare GLB exports from fork (DRTK) and official (nvdiffrast) implementations.
Generates GLB files with same seed and compares geometry, textures, and materials.
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

import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import struct

# Configuration
SEED = 42
IMAGE_PATH = "assets/example_image/T.png"
OUTPUT_DIR = Path("glb_comparison")
FORK_DIR = Path("/home/gaxxo/Documents/AI/TRELLIS.2-commercial-use")
OFFICIAL_DIR = Path("/home/gaxxo/Documents/AI/TRELLIS.2")

OUTPUT_DIR.mkdir(exist_ok=True)

def get_glb_info(glb_path):
    """Extract detailed info from GLB file."""
    import pygltflib
    
    glb = pygltflib.GLTF2().load(str(glb_path))
    
    info = {
        "file_path": str(glb_path),
        "file_size_bytes": glb_path.stat().st_size,
        "scene": {},
        "meshes": [],
        "materials": [],
        "textures": [],
        "images": [],
        "accessors": [],
        "buffer_views": [],
    }
    
    # Scene info
    if glb.scenes:
        for i, scene in enumerate(glb.scenes):
            info["scene"][f"scene_{i}"] = {
                "nodes": scene.nodes if scene.nodes else [],
                "name": scene.name,
            }
    
    # Mesh info
    if glb.meshes:
        for i, mesh in enumerate(glb.meshes):
            mesh_info = {
                "name": mesh.name,
                "primitives": []
            }
            for j, prim in enumerate(mesh.primitives):
                prim_info = {
                    "mode": prim.mode,
                    "indices": prim.indices,
                    "attributes": {},
                }
                if prim.attributes.POSITION is not None:
                    prim_info["attributes"]["POSITION"] = prim.attributes.POSITION
                if prim.attributes.NORMAL is not None:
                    prim_info["attributes"]["NORMAL"] = prim.attributes.NORMAL
                if prim.attributes.TEXCOORD_0 is not None:
                    prim_info["attributes"]["TEXCOORD_0"] = prim.attributes.TEXCOORD_0
                if prim.material is not None:
                    prim_info["material"] = prim.material
                mesh_info["primitives"].append(prim_info)
            info["meshes"].append(mesh_info)
    
    # Material info
    if glb.materials:
        for i, mat in enumerate(glb.materials):
            mat_info = {
                "name": mat.name,
                "pbr_metallic_roughness": None,
            }
            pbr = getattr(mat, 'pbrMetallicRoughness', None) or getattr(mat, 'pbr_metallic_roughness', None)
            if pbr:
                bct = getattr(pbr, 'baseColorTexture', None) or getattr(pbr, 'base_color_texture', None)
                mrt = getattr(pbr, 'metallicRoughnessTexture', None) or getattr(pbr, 'metallic_roughness_texture', None)
                bcf = getattr(pbr, 'baseColorFactor', None) or getattr(pbr, 'base_color_factor', None)
                mf = getattr(pbr, 'metallicFactor', None) or getattr(pbr, 'metallic_factor', None)
                rf = getattr(pbr, 'roughnessFactor', None) or getattr(pbr, 'roughness_factor', None)
                mat_info["pbr_metallic_roughness"] = {
                    "base_color_factor": list(bcf) if bcf else None,
                    "metallic_factor": mf,
                    "roughness_factor": rf,
                    "base_color_texture": bct.index if bct else None,
                    "metallic_roughness_texture": mrt.index if mrt else None,
                }
            info["materials"].append(mat_info)
    
    # Texture info
    if glb.textures:
        for i, tex in enumerate(glb.textures):
            tex_info = {
                "name": tex.name,
                "source": tex.source,
                "sampler": tex.sampler,
            }
            info["textures"].append(tex_info)
    
    # Image info
    if glb.images:
        for i, img in enumerate(glb.images):
            img_info = {
                "name": img.name,
                "uri": img.uri,
                "mimetype": img.mimeType,
                "buffer_view": img.bufferView,
            }
            info["images"].append(img_info)
    
    # Accessor info (vertex counts, etc.)
    if glb.accessors:
        for i, acc in enumerate(glb.accessors):
            acc_info = {
                "type": acc.type,
                "componentType": acc.componentType,
                "count": acc.count,
                "buffer_view": acc.bufferView,
            }
            info["accessors"].append(acc_info)
    
    return info

def get_mesh_stats(glb_path):
    """Extract mesh statistics (vertex count, face count, etc.)."""
    import pygltflib
    
    glb = pygltflib.GLTF2().load(str(glb_path))
    
    stats = {
        "total_vertices": 0,
        "total_faces": 0,
        "meshes": [],
    }
    
    for i, mesh in enumerate(glb.meshes):
        mesh_stats = {"name": mesh.name, "primitives": []}
        for j, prim in enumerate(mesh.primitives):
            prim_stats = {}
            
            # Get vertex count from POSITION accessor
            if prim.attributes.POSITION is not None:
                pos_acc = glb.accessors[prim.attributes.POSITION]
                prim_stats["vertices"] = pos_acc.count
                stats["total_vertices"] += pos_acc.count
            
            # Get face count from indices accessor
            if prim.indices is not None:
                idx_acc = glb.accessors[prim.indices]
                prim_stats["faces"] = idx_acc.count // 3
                stats["total_faces"] += idx_acc.count // 3
            
            mesh_stats["primitives"].append(prim_stats)
        stats["meshes"].append(mesh_stats)
    
    return stats

def extract_binary_data(glb_path):
    """Extract and decompress binary data from GLB."""
    import pygltflib
    
    glb = pygltflib.GLTF2().load(str(glb_path))
    
    data = {
        "vertices": None,
        "normals": None,
        "indices": None,
        "texcoords": None,
    }
    
    # Get buffer data
    try:
        buffer_data = glb.binary_buffer()
        if buffer_data is None:
            return data
    except:
        return data
    
    for mesh in glb.meshes:
        for prim in mesh.primitives:
            # Extract vertices
            if prim.attributes.POSITION is not None:
                acc = glb.accessors[prim.attributes.POSITION]
                bv = glb.bufferViews[acc.bufferView]
                data["vertices"] = np.frombuffer(
                    buffer_data[bv.byteOffset: bv.byteOffset + bv.byteLength],
                    dtype=np.float32
                ).reshape(-1, 3)
            
            # Extract normals
            if prim.attributes.NORMAL is not None:
                acc = glb.accessors[prim.attributes.NORMAL]
                bv = glb.bufferViews[acc.bufferView]
                data["normals"] = np.frombuffer(
                    buffer_data[bv.byteOffset: bv.byteOffset + bv.byteLength],
                    dtype=np.float32
                ).reshape(-1, 3)
            
            # Extract indices
            if prim.indices is not None:
                acc = glb.accessors[prim.indices]
                bv = glb.bufferViews[acc.bufferView]
                dtype = np.uint32 if acc.componentType == 5125 else np.uint16
                data["indices"] = np.frombuffer(
                    buffer_data[bv.byteOffset: bv.byteOffset + bv.byteLength],
                    dtype=dtype
                )
            
            # Extract texcoords
            if prim.attributes.TEXCOORD_0 is not None:
                acc = glb.accessors[prim.attributes.TEXCOORD_0]
                bv = glb.bufferViews[acc.bufferView]
                data["texcoords"] = np.frombuffer(
                    buffer_data[bv.byteOffset: bv.byteOffset + bv.byteLength],
                    dtype=np.float32
                ).reshape(-1, 2)
    
    return data

def extract_texture_data(glb_path, output_dir):
    """Extract embedded textures from GLB."""
    import pygltflib
    from PIL import Image
    import io
    
    glb = pygltflib.GLTF2().load(str(glb_path))
    textures = {}
    
    try:
        buffer_data = glb.binary_buffer()
        if buffer_data is None:
            return textures
    except:
        return textures
    
    for i, img in enumerate(glb.images):
        try:
            if img.bufferView is not None:
                bv = glb.bufferViews[img.bufferView]
                img_data = buffer_data[bv.byteOffset: bv.byteOffset + bv.byteLength]
                
                # Save extracted image
                img_path = output_dir / f"texture_{i}.png"
                pil_img = Image.open(io.BytesIO(img_data))
                pil_img.save(img_path)
                textures[f"texture_{i}"] = {
                    "path": str(img_path),
                    "size": pil_img.size,
                    "mode": pil_img.mode,
                }
        except Exception as e:
            print(f"  Warning: Could not extract texture {i}: {e}")
    
    return textures

def compare_glb_files(fork_path, official_path):
    """Compare two GLB files."""
    print("\n" + "="*80)
    print("GLB FILE COMPARISON")
    print("="*80)
    
    # File sizes
    fork_size = fork_path.stat().st_size
    official_size = official_path.stat().st_size
    size_diff = abs(fork_size - official_size)
    size_ratio = fork_size / official_size if official_size > 0 else 0
    
    print(f"\n[FILE SIZE]")
    print(f"  Fork:     {fork_size:,} bytes ({fork_size/1024/1024:.2f} MB)")
    print(f"  Official: {official_size:,} bytes ({official_size/1024/1024:.2f} MB)")
    print(f"  Diff:     {size_diff:,} bytes ({size_diff/1024:.2f} KB)")
    print(f"  Ratio:    {size_ratio:.4f}")
    
    # Mesh stats
    fork_stats = get_mesh_stats(fork_path)
    official_stats = get_mesh_stats(official_path)
    
    print(f"\n[MESH STATISTICS]")
    print(f"  Vertices:")
    print(f"    Fork:     {fork_stats['total_vertices']:,}")
    print(f"    Official: {official_stats['total_vertices']:,}")
    v_diff = abs(fork_stats['total_vertices'] - official_stats['total_vertices'])
    print(f"    Diff:     {v_diff:,} ({v_diff/official_stats['total_vertices']*100:.2f}%)")
    
    print(f"  Faces:")
    print(f"    Fork:     {fork_stats['total_faces']:,}")
    print(f"    Official: {official_stats['total_faces']:,}")
    f_diff = abs(fork_stats['total_faces'] - official_stats['total_faces'])
    print(f"    Diff:     {f_diff:,} ({f_diff/official_stats['total_faces']*100:.2f}%)")
    
    # Materials
    fork_info = get_glb_info(fork_path)
    official_info = get_glb_info(official_path)
    
    print(f"\n[MATERIALS]")
    print(f"  Fork:     {len(fork_info['materials'])} materials")
    print(f"  Official: {len(official_info['materials'])} materials")
    
    for i, (fm, om) in enumerate(zip(fork_info['materials'], official_info['materials'])):
        print(f"\n  Material {i}:")
        if fm['pbr_metallic_roughness'] and om['pbr_metallic_roughness']:
            fpbr = fm['pbr_metallic_roughness']
            opbr = om['pbr_metallic_roughness']
            print(f"    Base color factor: fork={fpbr['base_color_factor']}, official={opbr['base_color_factor']}")
            print(f"    Metallic factor:   fork={fpbr['metallic_factor']}, official={opbr['metallic_factor']}")
            print(f"    Roughness factor:  fork={fpbr['roughness_factor']}, official={opbr['roughness_factor']}")
    
    # Textures
    print(f"\n[TEXTURES]")
    print(f"  Fork:     {len(fork_info['textures'])} textures, {len(fork_info['images'])} images")
    print(f"  Official: {len(official_info['textures'])} textures, {len(official_info['images'])} images")
    
    # Extract and compare textures
    fork_tex_dir = OUTPUT_DIR / "fork_textures"
    official_tex_dir = OUTPUT_DIR / "official_textures"
    fork_tex_dir.mkdir(exist_ok=True)
    official_tex_dir.mkdir(exist_ok=True)
    
    fork_textures = extract_texture_data(fork_path, fork_tex_dir)
    official_textures = extract_texture_data(official_path, official_tex_dir)
    
    print(f"\n  Extracted textures:")
    for name in set(fork_textures.keys()) | set(official_textures.keys()):
        ft = fork_textures.get(name)
        ot = official_textures.get(name)
        if ft and ot:
            print(f"    {name}:")
            print(f"      Fork size:     {ft['size']}")
            print(f"      Official size: {ot['size']}")
            if ft['size'] == ot['size']:
                # Compare pixel data
                from PIL import Image
                fork_img = Image.open(ft['path'])
                official_img = Image.open(ot['path'])
                f_arr = np.array(fork_img)
                o_arr = np.array(official_img)
                diff = np.abs(f_arr.astype(float) - o_arr.astype(float))
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                print(f"      Mean pixel diff: {mean_diff:.2f}")
                print(f"      Max pixel diff:  {max_diff}")
        else:
            print(f"    {name}: Only in {'fork' if ft else 'official'}")
    
    # Binary data comparison
    fork_data = extract_binary_data(fork_path)
    official_data = extract_binary_data(official_path)
    
    print(f"\n[GEOMETRY DATA]")
    if fork_data['vertices'] is not None and official_data['vertices'] is not None:
        v_diff = np.abs(fork_data['vertices'] - official_data['vertices'])
        print(f"  Vertices shape:      fork={fork_data['vertices'].shape}, official={official_data['vertices'].shape}")
        print(f"  Vertex mean diff:   {np.mean(v_diff):.6f}")
        print(f"  Vertex max diff:     {np.max(v_diff):.6f}")
    
    if fork_data['normals'] is not None and official_data['normals'] is not None:
        n_diff = np.abs(fork_data['normals'] - official_data['normals'])
        print(f"  Normals shape:      fork={fork_data['normals'].shape}, official={official_data['normals'].shape}")
        print(f"  Normal mean diff:    {np.mean(n_diff):.6f}")
        print(f"  Normal max diff:     {np.max(n_diff):.6f}")
        # Check for zero-length normals
        fork_norms = np.linalg.norm(fork_data['normals'], axis=1)
        official_norms = np.linalg.norm(official_data['normals'], axis=1)
        fork_zero = np.sum(fork_norms < 0.99)
        official_zero = np.sum(official_norms < 0.99)
        print(f"  Non-unit normals:    fork={fork_zero}, official={official_zero}")
    
    if fork_data['indices'] is not None and official_data['indices'] is not None:
        print(f"  Indices shape:       fork={fork_data['indices'].shape}, official={official_data['indices'].shape}")
        indices_match = np.array_equal(fork_data['indices'], official_data['indices'])
        print(f"  Indices match:       {indices_match}")
    
    print("\n" + "="*80)
    
    return {
        "fork_size": fork_size,
        "official_size": official_size,
        "fork_stats": fork_stats,
        "official_stats": official_stats,
        "fork_textures": fork_textures,
        "official_textures": official_textures,
    }

def main():
    # Check if both GLB files exist
    fork_glb = OUTPUT_DIR / "fork_sample.glb"
    official_glb = OUTPUT_DIR / "official_sample.glb"
    
    if not fork_glb.exists():
        print(f"ERROR: Fork GLB not found at {fork_glb}")
        print("Run inference first:")
        print(f"  cd {FORK_DIR} && conda run -n trellis2 python example_glb.py")
        sys.exit(1)
    
    if not official_glb.exists():
        print(f"ERROR: Official GLB not found at {official_glb}")
        print("Run inference first:")
        print(f"  cd {OFFICIAL_DIR} && conda run -n trellis2-official python example_glb.py")
        sys.exit(1)
    
    # Compare GLB files
    results = compare_glb_files(fork_glb, official_glb)
    
    # Save results
    results_path = OUTPUT_DIR / "comparison_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            "fork_size": results["fork_size"],
            "official_size": results["official_size"],
            "fork_stats": results["fork_stats"],
            "official_stats": results["official_stats"],
        }
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()