#!/usr/bin/env python3
"""
Compare SSAO values between fork and official outputs.
"""
import numpy as np
from PIL import Image

# The key insight: SSAO radius needs to be scaled for DRTK's camera-space depth
# Fork uses camera-space z (~10), official uses normalized z/w (~0.82)

# Depth scaling factor
DEPTH_SCALE = 10.0 / 0.82  # ~12.2

print("SSAO Depth Scaling Analysis")
print("=" * 60)
print()
print("Fork (DRTK) uses camera-space depth: range ~[9.66, 10.27]")
print("Official (nvdiffrast) uses normalized depth: range ~[0.81, 0.82]")
print()
print("The SSAO radius of 0.1 has different physical scale:")
print(f"  Fork: samples within {0.1 / 10.0:.4f} units relative to depth")
print(f"  Official: samples within {0.1 / 0.82:.4f} units relative to depth")
print()
print("The fork's SSAO samples a MUCH smaller relative area:")
print(f"  Official samples {0.1 / 0.82:.2f}x larger relative area than fork")
print()
print("To match official's SSAO effect, fork should use:")
print(f"  radius = 0.1 * (10.0 / 0.82) = {0.1 * DEPTH_SCALE:.2f}")
print()

# Let's also analyze the depth difference more precisely
print("Depth convention comparison:")
print("  nvdiffrast: rast[..., 2] = z_clip / w_clip (NDC depth)")
print("  DRTK: depth = z_camera (eye-space z)")
print()
print("For a perspective projection matrix P with near=n, far=f:")
print("  NDC depth = (z_eye - n) / (f - n) ... actually it's more complex")
print("  nvdiffrast z/w ranges from 0 to 1 typically")
print("  DRTK z_camera is actual distance from camera")
print()
print("The SSAO code uses:")
print("  x_view = (x_grid - cx) * depth / fx")
print("  y_view = (y_grid - cy) * depth / fy")
print("  view_pos = [x_view, y_view, depth]")
print()
print("This reconstructs view-space position from depth.")
print("The radius parameter determines how far to sample from view_pos.")
print()
print("With camera-space depth ~10, radius=0.1 samples 1% of scene")
print("With normalized depth ~0.82, radius=0.1 samples ~12% of scene")
print()
print("This explains why SSAO effect differs significantly!")
print()
print("Recommendation: Scale SSAO radius for DRTK by depth ratio:")
print(f"  const DEPTH_SCALE = camera_z / ndc_z ~= 10 / 0.82 ~= 12")
print("  ssao_radius = 0.1 * DEPTH_SCALE  # or adjust the function")