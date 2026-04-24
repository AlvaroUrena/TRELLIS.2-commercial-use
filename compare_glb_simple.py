#!/usr/bin/env python
"""
Compare GLB exports from fork (DRTK) and official (nvdiffrast) implementations.
Simple version using trimesh and basic file operations.
"""
import sys
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("glb_comparison")

def main():
    fork_glb = OUTPUT_DIR / "fork_sample.glb"
    official_glb = OUTPUT_DIR / "official_sample.glb"
    
    print("=" * 80)
    print("GLB FILE COMPARISON")
    print("=" * 80)
    
    # File sizes
    fork_size = fork_glb.stat().st_size
    official_size = official_glb.stat().st_size
    size_diff = abs(fork_size - official_size)
    size_ratio = fork_size / official_size if official_size > 0 else 0
    
    print(f"\n[FILE SIZE]")
    print(f"  Fork:     {fork_size:,} bytes ({fork_size/1024/1024:.2f} MB)")
    print(f"  Official: {official_size:,} bytes ({official_size/1024/1024:.2f} MB)")
    print(f"  Diff:     {size_diff:,} bytes ({size_diff/1024:.2f} KB)")
    print(f"  Ratio:    {size_ratio:.4f}")
    
    # Use trimesh to compare meshes
    import trimesh
    
    print(f"\n[MESH STATISTICS]")
    
    # Load meshes
    try:
        fork_scene = trimesh.load(str(fork_glb))
        official_scene = trimesh.load(str(official_glb))
        
        # Get geometry
        fork_meshes = list(fork_scene.geometry.values()) if hasattr(fork_scene, 'geometry') else [fork_scene]
        official_meshes = list(official_scene.geometry.values()) if hasattr(official_scene, 'geometry') else [official_scene]
        
        fork_vertices = sum(m.vertices.shape[0] for m in fork_meshes if hasattr(m, 'vertices'))
        fork_faces = sum(m.faces.shape[0] for m in fork_meshes if hasattr(m, 'faces'))
        official_vertices = sum(m.vertices.shape[0] for m in official_meshes if hasattr(m, 'vertices'))
        official_faces = sum(m.faces.shape[0] for m in official_meshes if hasattr(m, 'faces'))
        
        print(f"  Vertices:")
        print(f"    Fork:     {fork_vertices:,}")
        print(f"    Official: {official_vertices:,}")
        v_diff = abs(fork_vertices - official_vertices)
        print(f"    Diff:     {v_diff:,} ({v_diff/official_vertices*100:.2f}%)")
        
        print(f"  Faces:")
        print(f"    Fork:     {fork_faces:,}")
        print(f"    Official: {official_faces:,}")
        f_diff = abs(fork_faces - official_faces)
        print(f"    Diff:     {f_diff:,} ({f_diff/official_faces*100:.2f}%)")
        
        # Compare vertex bounding boxes
        if fork_meshes and official_meshes:
            fork_bounds = np.array([m.bounds for m in fork_meshes if hasattr(m, 'bounds')])
            official_bounds = np.array([m.bounds for m in official_meshes if hasattr(m, 'bounds')])
            
            if len(fork_bounds) > 0 and len(official_bounds) > 0:
                fork_min = fork_bounds[:, 0].min(axis=0)
                fork_max = fork_bounds[:, 1].max(axis=0)
                official_min = official_bounds[:, 0].min(axis=0)
                official_max = official_bounds[:, 1].max(axis=0)
                
                print(f"\n  Bounding Box:")
                print(f"    Fork min:     [{fork_min[0]:.4f}, {fork_min[1]:.4f}, {fork_min[2]:.4f}]")
                print(f"    Official min: [{official_min[0]:.4f}, {official_min[1]:.4f}, {official_min[2]:.4f}]")
                print(f"    Fork max:     [{fork_max[0]:.4f}, {fork_max[1]:.4f}, {fork_max[2]:.4f}]")
                print(f"    Official max: [{official_max[0]:.4f}, {official_max[1]:.4f}, {official_max[2]:.4f}]")
        
        # Check for valid normals
        print(f"\n[NORMALS CHECK]")
        for i, m in enumerate(fork_meshes):
            if hasattr(m, 'vertex_normals') and m.vertex_normals is not None:
                norms = np.linalg.norm(m.vertex_normals, axis=1)
                non_unit = np.sum(norms < 0.99) + np.sum(norms > 1.01)
                print(f"  Fork mesh {i}: {non_unit} non-unit normals")
                if non_unit > 0:
                    bad_norms = np.where((norms < 0.99) | (norms > 1.01))[0]
                    print(f"    First 5 non-unit magnitudes: {norms[bad_norms[:5]]}")
        
        for i, m in enumerate(official_meshes):
            if hasattr(m, 'vertex_normals') and m.vertex_normals is not None:
                norms = np.linalg.norm(m.vertex_normals, axis=1)
                non_unit = np.sum(norms < 0.99) + np.sum(norms > 1.01)
                print(f"  Official mesh {i}: {non_unit} non-unit normals")
        
        # Compare visual hashes if available
        print(f"\n[VISUAL COMPARISON]")
        print(f"  GLB files ready for manual inspection:")
        print(f"  Fork:     {fork_glb}")
        print(f"  Official: {official_glb}")
        
    except Exception as e:
        print(f"  Error loading meshes: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Open both GLB files in glTF viewer: https://gltf-viewer.donmccurdy.com/")
    print("2. Load in Blender to compare visual quality")
    print("3. Check for any glTF validation errors")

if __name__ == "__main__":
    main()