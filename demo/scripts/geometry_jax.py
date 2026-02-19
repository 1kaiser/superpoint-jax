import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['shape'])
def radial_confidence_mask(shape):
    """
    Generates a radial confidence mask.
    Center 70% of the image has high weight (1.0), falling off to 0.2 at the periphery.

    Args:
        shape: Tuple (H, W).

    Returns:
        jnp.ndarray: Weight mask of shape (H, W).
    """
    H, W = shape
    y = jnp.arange(H, dtype=jnp.float32)
    x = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(y, x, indexing='ij')

    cy, cx = H / 2.0, W / 2.0

    # Normalized radius from center [0, 1] (1 being the corner)
    # Using Euclidean distance
    dist = jnp.sqrt((yy - cy)**2 + (xx - cx)**2)
    max_dist = jnp.sqrt(cy**2 + cx**2)
    norm_dist = dist / max_dist

    # "Center 70%": Let's interpret this as the area within 0.7 * max_radius?
    # Or simply 0.7 of the pixels?
    # Let's map radius 0 -> 1.0, radius 1 -> 0.2.
    # Linear: 1.0 - 0.8 * r

    # However, "Center 70%: High Weight (1.0)" might mean a plateau.
    # Let's use a plateau until r=0.5, then decay?
    # Simpler interpretation: Linear decay from center to corner is robust.
    # weight = 1.0 - 0.8 * norm_dist

    weight = 1.0 - 0.8 * norm_dist
    return jnp.clip(weight, 0.2, 1.0)

@partial(jax.jit, static_argnames=['shape', 'border_percent'])
def boundary_mask(shape, border_percent=0.10):
    """
    Creates a binary mask that discards the outer `border_percent` of the image.

    Args:
        shape: Tuple (H, W).
        border_percent: Float, fraction of image to discard from each side.

    Returns:
        jnp.ndarray: Binary mask (0 or 1) of shape (H, W).
    """
    H, W = shape
    h_border = int(H * border_percent)
    w_border = int(W * border_percent)

    mask = jnp.zeros(shape, dtype=jnp.float32)
    mask = mask.at[h_border:H-h_border, w_border:W-w_border].set(1.0)

    return mask

@partial(jax.jit)
def depth_edge_mask(depth_map, threshold=0.1):
    """
    Computes a mask to discard high-gradient depth edges (silhouette halos).

    Args:
        depth_map: Depth image (H, W).
        threshold: Gradient magnitude threshold to consider an edge.

    Returns:
        jnp.ndarray: Binary mask (1 for smooth regions, 0 for edges).
    """
    # Simple Sobel-like gradients
    dy = depth_map[1:, :] - depth_map[:-1, :]
    dx = depth_map[:, 1:] - depth_map[:, :-1]

    # Pad back to original size
    dy = jnp.pad(dy, ((0, 1), (0, 0)), mode='constant')
    dx = jnp.pad(dx, ((0, 0), (0, 1)), mode='constant')

    grad_mag = jnp.sqrt(dy**2 + dx**2)

    # Invert: Low gradient -> 1, High gradient -> 0
    # We can use a soft threshold or hard threshold.
    # Hard threshold:
    mask = (grad_mag < threshold).astype(jnp.float32)

    return mask

@partial(jax.jit)
def lift_to_3d(keypoints, depth_values, intrinsics):
    """
    Lifts 2D keypoints to 3D using depth values and intrinsics.

    Args:
        keypoints: (N, 2) array of (x, y) coordinates.
        depth_values: (N,) array of depth values at keypoints.
        intrinsics: (3, 3) calibration matrix.

    Returns:
        jnp.ndarray: (N, 3) array of 3D points.
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = keypoints[:, 0]
    v = keypoints[:, 1]

    z = depth_values
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return jnp.stack([x, y, z], axis=-1)

@partial(jax.jit)
def kabsch_weighted(points_src, points_dst, weights):
    """
    Computes the optimal rigid transformation (R, t) aligning points_src to points_dst
    minimizing the weighted RMSD.

    Args:
        points_src: (N, 3) source points.
        points_dst: (N, 3) destination points.
        weights: (N,) weights.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
    """
    # Normalize weights
    weights_norm = weights / (jnp.sum(weights) + 1e-6)

    # Compute weighted centroids
    centroid_src = jnp.sum(points_src * weights_norm[:, None], axis=0)
    centroid_dst = jnp.sum(points_dst * weights_norm[:, None], axis=0)

    # Center the points
    src_centered = points_src - centroid_src
    dst_centered = points_dst - centroid_dst

    # Compute weighted covariance matrix
    # H = src_centered.T @ W @ dst_centered
    H = (src_centered.T * weights_norm) @ dst_centered

    # SVD
    U, S, Vt = jnp.linalg.svd(H)
    V = Vt.T

    # Rotation
    d = jnp.linalg.det(V @ U.T)

    # Handle reflection case
    # diag = jnp.array([1, 1, d])
    # R = V @ jnp.diag(diag) @ U.T

    # JAX compatible diagonal construction
    diag_val = jnp.ones(3).at[2].set(d)
    R = V @ jnp.diag(diag_val) @ U.T

    # Translation
    t = centroid_dst - R @ centroid_src

    return R, t

@partial(jax.jit)
def compute_reprojection_error(points_src, points_dst, R, t):
    """
    Computes the Euclidean distance between transformed source points and destination points.

    Args:
        points_src: (N, 3)
        points_dst: (N, 3)
        R: (3, 3)
        t: (3,)

    Returns:
        jnp.ndarray: (N,) errors.
    """
    transformed = (points_src @ R.T) + t
    diff = transformed - points_dst
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))

@partial(jax.jit)
def apply_transform(points, R, t):
    """
    Applies R, t to points.

    Args:
        points: (N, 3)
        R: (3, 3)
        t: (3,)

    Returns:
        (N, 3) transformed points.
    """
    return (points @ R.T) + t
