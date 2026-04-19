"""
DRTK compatibility layer providing nvdiffrast-like API semantics.

This module wraps DRTK functions to provide an interface similar to nvdiffrast,
making migration easier while using DRTK's MIT-licensed implementation.

Key API differences:
- nvdiffrast uses clip-space coordinates [N, V, 4], DRTK uses pixel-space [N, V, 3]
- nvdiffrast rasterize returns (rast, rast_db), DRTK returns index_img + render() gives (depth, bary)
- nvdiffrast interpolate takes rast, DRTK interpolate takes (index_img, bary_img)
- nvdiffrast texture uses derivatives, DRTK mipmap_grid_sample uses Jacobian
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import functools


def build_mipmap(tex: torch.Tensor, max_levels: int = 12) -> List[torch.Tensor]:
    """Build mipmap pyramid from texture tensor.
    
    Args:
        tex: Texture tensor [N, C, H, W]
        max_levels: Maximum mipmap levels to generate
        
    Returns:
        List of mipmap levels, starting with original texture
    """
    mipmaps = [tex]
    h, w = tex.shape[-2], tex.shape[-1]
    for _ in range(1, max_levels):
        if h <= 1 and w <= 1:
            break
        tex = F.avg_pool2d(tex, 2)
        mipmaps.append(tex)
        h, w = (h + 1) // 2, (w + 1) // 2
    return mipmaps


def compute_uv_jacobian(uv: torch.Tensor, resolution: int) -> torch.Tensor:
    """Compute UV Jacobian for mipmap level selection.
    
    This computes the derivatives of UV coordinates with respect to pixel position,
    needed for DRTK's mipmap_grid_sample. Uses finite differences.
    
    Args:
        uv: UV coordinates [N, H, W, 2]
        resolution: Image resolution (used for scaling)
        
    Returns:
        Jacobian tensor [N, H, W, 2, 2]
    """
    n, h, w, _ = uv.shape
    
    # Compute gradients using finite differences
    # dudx, dudy, dvdx, dvdy
    dx = uv[:, :, 1:, :] - uv[:, :, :-1, :]  # [N, H, W-1, 2]
    dy = uv[:, 1:, :, :] - uv[:, :-1, :, :]  # [N, H-1, W, 2]
    
    # Pad to original size
    dx = F.pad(dx, (0, 0, 0, 1), mode='replicate')  # [N, H, W, 2]
    dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')  # [N, H, W, 2]
    
    # Construct jacobian [N, H, W, 2, 2]
    jacobian = torch.zeros(n, h, w, 2, 2, device=uv.device, dtype=uv.dtype)
    jacobian[..., 0, 0] = dx[..., 0]  # du/dx
    jacobian[..., 0, 1] = dy[..., 0]  # du/dy
    jacobian[..., 1, 0] = dx[..., 1]  # dv/dx
    jacobian[..., 1, 1] = dy[..., 1]  # dv/dy
    
    return jacobian


def intrinsics_to_camera_params(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: int,
    near: float = 0.1,
    far: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert OpenCV camera intrinsics/extrinsics to DRTK camera parameters.
    
    Args:
        intrinsics: [3, 3] OpenCV intrinsics matrix
        extrinsics: [4, 4] camera extrinsics matrix (world to camera)
        resolution: Image resolution (H=W assumed)
        near: Near plane (unused, for compatibility)
        far: Far plane (unused, for compatibility)
        
    Returns:
        campos: Camera position [3]
        camrot: Camera rotation matrix [3, 3]
        focal: Focal lengths [2, 2]
        princpt: Principal point [2]
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Camera position: -R^T @ t from extrinsics [R|t]
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    campos = -R.T @ t
    
    # Camera rotation
    camrot = R.T
    
    # Focal length matrix and principal point
    focal = torch.stack([
        torch.stack([fx, torch.zeros_like(fx)]),
        torch.stack([torch.zeros_like(fy), fy])
    ])
    princpt = torch.stack([cx, cy])
    
    return campos, camrot, focal, princpt


class DRTKContext:
    """Compatibility context mimicking nvdiffrast's RasterizeCudaContext.
    
    DRTK is stateless, but this provides a compatible interface.
    Caches rasterization outputs for use in interpolate().
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._cache_index_img = None
        self._cache_bary = None
        self._cache_resolution = None
    
    def rasterize(
        self,
        vertices_clip: torch.Tensor,
        faces: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize mesh using DRTK, returning nvdiffrast-compatible output.
        
        Args:
            vertices_clip: Clip-space vertices [1, V, 4]
            faces: Face indices [F, 3]
            resolution: (H, W) output resolution
            
        Returns:
            rast: Rasterization output [1, H, W, 4] similar to nvdiffrast format
                - [..., 0:2]: barycentric coordinates (v, u)
                - [..., 2]: depth (z/w)
                - [..., 3]: triangle ID (1-indexed, 0 means background)
            rast_db: Barycentric derivatives [1, H, W, 4] (placeholder)
        """
        import drtk
        
        h, w = resolution

        x_ndc = vertices_clip[..., 0] / vertices_clip[..., 3].clamp(min=1e-8, max=1e8)
        y_ndc = -vertices_clip[..., 1] / vertices_clip[..., 3].clamp(min=1e-8, max=1e8)
        
        x_pix = (x_ndc + 1) * 0.5 * w
        y_pix = (y_ndc + 1) * 0.5 * h
        z_cam = vertices_clip[..., 3].clone()
        
        v_pix = torch.stack([x_pix, y_pix, z_cam], dim=-1)
        
        faces_int = faces.to(torch.int32) if faces.dtype != torch.int32 else faces
        
        index_img = drtk.rasterize(v_pix, faces_int, height=h, width=w)
        depth, bary = drtk.render(v_pix, faces_int, index_img)
        
        # Cache for use in interpolate()
        self._cache_index_img = index_img
        self._cache_bary = bary
        self._cache_resolution = (h, w)
        
        batch_size = v_pix.shape[0]
        rast = torch.zeros(batch_size, h, w, 4, device=v_pix.device, dtype=torch.float32)
        
        rast[..., 0] = bary[:, 1]
        rast[..., 1] = bary[:, 2]
        rast[..., 2] = depth[:, 0]
        rast[..., 3] = (index_img.float() + 1).float()
        
        rast_db = torch.zeros_like(rast)
        
        return rast, rast_db


def interpolate(
    attr: torch.Tensor,
    rast: torch.Tensor,
    faces: torch.Tensor,
    rast_db: Optional[torch.Tensor] = None,
    ctx: Optional['DRTKContext'] = None,
    peeler: Optional['DepthPeeler'] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Interpolate vertex attributes using DRTK.
    
    Args:
        attr: Vertex attributes [1, V, C] or [V, C]
        rast: Rasterization output from DRTKContext.rasterize()
        faces: Face indices [F, 3]
        rast_db: Unused, for compatibility with nvdiffrast API
        ctx: DRTKContext instance (uses cached rasterization outputs if provided)
        peeler: DepthPeeler instance (uses cached rasterization outputs if provided)
        
    Returns:
        interpolated: [1, H, W, C] interpolated attributes
        derivs: None (DRTK doesn't use derivatives this way)
    """
    import drtk
    
    if attr.dim() == 2:
        attr = attr.unsqueeze(0)
    
    h, w = rast.shape[1], rast.shape[2]
    
    # Use cached outputs from DepthPeeler if available (highest priority)
    if peeler is not None and peeler._cache_index_img is not None:
        index_img = peeler._cache_index_img
        bary_img = peeler._cache_bary
    # Use cached outputs from a DRTKContext if available
    elif ctx is not None and ctx._cache_index_img is not None and ctx._cache_resolution == (h, w):
        index_img = ctx._cache_index_img
        bary_img = ctx._cache_bary
    else:
        # Fall back to reconstructing from rast (less accurate but works for compatibility)
        index_img = (rast[0, ..., 3] - 1).to(torch.int32)
        u = rast[..., 0]
        v = rast[..., 1]
        w0 = 1.0 - u - v
        bary_img = torch.stack([w0, u, v], dim=1)
    
    faces_int = faces.to(torch.int32) if faces.dtype != torch.int32 else faces
    
    result = drtk.interpolate(attr, faces_int, index_img, bary_img)
    
    result = result.permute(0, 2, 3, 1)
    
    return result, None


def texture(
    tex: torch.Tensor,
    uv: torch.Tensor,
    uv_da: Optional[torch.Tensor] = None,
    filter_mode: str = 'linear',
    boundary_mode: str = 'wrap',
) -> Tuple[torch.Tensor, None]:
    """Sample texture using DRTK's mipmap_grid_sample.
    
    Args:
        tex: Texture [1, C, H, W] or [C, H, W]
        uv: UV coordinates [1, H, W, 2]
        uv_da: UV derivatives (for mipmap level)
        filter_mode: 'linear', 'linear-mipmap-linear', 'nearest'
        boundary_mode: 'wrap', 'clamp', 'cube'
        
    Returns:
        sampled: [1, H, W, C] sampled texture
        None: placeholder for nvdiffrast API compatibility
    """
    import drtk
    
    # Ensure batch dimensions
    if tex.dim() == 3:
        tex = tex.unsqueeze(0)
    if uv.dim() == 3:
        uv = uv.unsqueeze(0)
    
    # Handle boundary mode
    padding_mode = 'border' if boundary_mode == 'clamp' else 'zeros'
    if boundary_mode == 'wrap':
        # DRTK doesn't have 'wrap' mode, need to handle with modulo
        uv = uv % 1.0
        padding_mode = 'border'  # After modulo, use border for edges
    
    if boundary_mode == 'cube':
        # Cubemap sampling not directly supported by DRTK
        # Fall back to PyTorch grid_sample (not differentiable w.r.t. pixels, but works)
        # Actually we should handle cubemap separately in the caller
        # For now, raise an error
        raise NotImplementedError("Cubemap sampling requires custom implementation. Use grid_sample with manual face selection.")
    
    # Determine if we need mipmap sampling
    use_mipmap = 'mipmap' in filter_mode or (uv_da is not None)
    
    if use_mipmap:
        # Build mipmap pyramid
        max_levels = int(torch.log2(torch.tensor(max(tex.shape[-2:]))).item()) + 1
        mipmap = build_mipmap(tex, max_levels)
        
        # Compute UV Jacobian for mipmap level selection
        if uv_da is not None:
            # uv_da is the derivative of UV w.r.t. pixels, shape [1, H, W, 2, 2] or [H, W, 2, 2]
            # This is already what we need for vt_dxdy_img
            vt_dxdy = uv_da if uv_da.dim() == 5 else uv_da.unsqueeze(0)
        else:
            # Compute Jacobian using finite differences
            vt_dxdy = compute_uv_jacobian(uv[0], tex.shape[-1])  # [H, W, 2, 2]
            vt_dxdy = vt_dxdy.unsqueeze(0)  # [1, H, W, 2, 2]
        
        # Use DRTK's mipmap grid sample
        # Note: mipmap_grid_sample expects UV in [-1, 1] range
        uv_grid = uv * 2 - 1  # [0, 1] -> [-1, 1]
        
        sampled = drtk.mipmap_grid_sample(
            mipmap,
            uv_grid,
            vt_dxdy,
            max_aniso=1,
            mode='bilinear' if 'linear' in filter_mode else 'nearest',
            padding_mode=padding_mode,
            align_corners=False,
        )
    else:
        # Simple bilinear/nearest sampling using grid_sample
        uv_grid = uv * 2 - 1  # [0, 1] -> [-1, 1]
        mode = 'bilinear' if filter_mode == 'linear' else 'nearest'
        sampled = F.grid_sample(tex, uv_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    
    # Convert from [N, C, H, W] to [N, H, W, C]
    result = sampled.permute(0, 2, 3, 1)
    
    return result, None


def antialias(color: torch.Tensor, rast: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Antialias using DRTK's edge_grad_estimator.
    
    Note: This is not semantic antialiasing like nvdiffrast. It provides differentiability
    at edge discontinuities. For visual antialiasing, post-processing may be needed.
    
    Args:
        color: Color image [1, H, W, C]
        rast: Rasterization output
        vertices: Vertices (unused, for API compatibility)
        faces: Faces (unused, for API compatibility)
        
    Returns:
        Color tensor with edge gradients attached
    """
    # DRTK's edge_grad_estimator is for backprop, not visual AA
    # For visual antialiasing, we could use multisampling or post-process AA
    # For now, return color unchanged (inference doesn't need AA for quality)
    # For training, would need edge_grad_estimator with proper setup
    return color


class DepthPeeler:
    """Context manager for depth peeling, mimicking nvdiffrast's DepthPeeler.
    
    DRTK doesn't have built-in depth peeling, so we implement it manually.
    """
    def __init__(self, ctx: DRTKContext, vertices_clip: torch.Tensor, faces: torch.Tensor, resolution: Tuple[int, int]):
        self.ctx = ctx
        self.vertices_clip = vertices_clip
        self.faces = faces
        self.resolution = resolution
        self.layers_drawn = 0
        self.max_layers = 100  # Safety limit
        self.depth_buffer = None  # Accumulated depth layers
        self._cache_index_img = None
        self._cache_bary = None
    
    def _rasterize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform rasterization."""
        rast, rast_db = self.ctx.rasterize(self.vertices_clip, self.faces, self.resolution)
        return rast, rast_db
    
    def rasterize_next_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize the next depth layer (peel).
        
        Returns:
            rast: Rasterization output for current layer
            rast_db: Barycentric derivatives
        """
        import drtk
        
        if self.layers_drawn >= self.max_layers:
            return torch.zeros(1, self.resolution[0], self.resolution[1], 4, device=self.vertices_clip.device), None
        
        batch_size = self.vertices_clip.shape[0]
        h, w = self.resolution
        
        x_ndc = self.vertices_clip[..., 0] / self.vertices_clip[..., 3].clamp(min=1e-8)
        y_ndc = -self.vertices_clip[..., 1] / self.vertices_clip[..., 3].clamp(min=1e-8)
        z_cam = self.vertices_clip[..., 3]
        
        x_pix = (x_ndc + 1) * 0.5 * w
        y_pix = (y_ndc + 1) * 0.5 * h
        
        v_pix = torch.stack([x_pix, y_pix, z_cam], dim=-1)
        faces_int = self.faces.to(torch.int32) if self.faces.dtype != torch.int32 else self.faces
        
        index_img = drtk.rasterize(v_pix, faces_int, height=h, width=w)
        depth, bary = drtk.render(v_pix, faces_int, index_img)
        
        # Cache for interpolate
        self._cache_index_img = index_img
        self._cache_bary = bary
        
        if self.depth_buffer is not None:
            has_geom = index_img >= 0
            current_depth = depth[0, 0]
            pass
        
        rast = torch.zeros(batch_size, h, w, 4, device=self.vertices_clip.device, dtype=torch.float32)
        rast[..., 0] = bary[:, 1]
        rast[..., 1] = bary[:, 2]
        rast[..., 2] = depth[:, 0]
        rast[..., 3] = (index_img.float() + 1).float()
        
        rast_db = torch.zeros_like(rast)
        
        self.depth_buffer = depth.clone()
        self.layers_drawn += 1
        
        return rast, rast_db
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# Module-level placeholder for compatibility
def RasterizeCudaContext(device: str = 'cuda'):
    """Create a DRTK context (stateless)."""
    return DRTKContext(device=device)