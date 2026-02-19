import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import cv2
import trimesh
import torch
from pathlib import Path
from tqdm import tqdm

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import local modules
from demo.scripts.geometry_jax import (
    radial_confidence_mask,
    boundary_mask,
    depth_edge_mask,
    lift_to_3d,
    kabsch_weighted,
    compute_reprojection_error
)
from demo.scripts.tracker import PersistenceTracker
from demo.scripts.export_utils import export_to_las, export_to_glb
from superpoint_jax.model.superpoint_jax import SuperPointJAX
from superpoint_jax.model.lightglue_jax import LightGlueJAX
from superpoint_jax.utils.convert_to_jax import convert_superpoint_weights, convert_lightglue_weights
from superpoint_jax.model.superpoint_torch import SuperPointTorch
from lightglue import LightGlue

# Constants
INPUT_DIR = ROOT_DIR / 'input_data/input_frames'
DEPTH_DIR = ROOT_DIR / 'input_data/depth_maps'
WEIGHTS_DIR = ROOT_DIR / 'weights'
OUTPUT_DIR = ROOT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    print("Loading models...")

    # SuperPoint
    # Load PyTorch first (as per existing workflow)
    sp_torch = SuperPointTorch(max_num_keypoints=1024).to(device)
    sp_torch.load_state_dict(torch.load(str(WEIGHTS_DIR / 'superpoint_torch.pth'), map_location=device))
    sp_torch.eval()

    # Convert to JAX
    sp_jax = SuperPointJAX(max_num_keypoints=1024, detection_threshold=0.015, rngs=nnx.Rngs(0))
    sp_jax = convert_superpoint_weights(sp_torch, sp_jax)

    # LightGlue
    lg_torch = LightGlue(features='superpoint').to(device)
    lg_torch.eval()

    lg_jax = LightGlueJAX(rngs=nnx.Rngs(0))
    lg_jax = convert_lightglue_weights(lg_torch, lg_jax)

    print("Models loaded and converted to JAX.")
    return sp_jax, lg_jax

def load_image_pair(idx, frames):
    img_name = frames[idx]
    img_path = INPUT_DIR / img_name
    depth_path = DEPTH_DIR / img_name  # Assuming same name

    if not depth_path.exists():
        pass

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH) # 16-bit

    if img is None or depth is None:
        return None, None, None

    img_float = img.astype(np.float32) / 255.0

    # Check depth scale. Usually millimeters or arbitrary.
    # Assuming standard depth maps where value is depth in mm or similar unit.
    # We'll normalize or use as is. Let's assume mm and convert to meters.
    depth_float = depth.astype(np.float32) / 1000.0

    return img_float, depth_float, img_name

def run_pipeline():
    sp_jax, lg_jax = load_models()

    frames = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    if not frames:
        print("No frames found!")
        return

    # Intrinsics (Approximate)
    # Assume 640x480
    test_img = cv2.imread(str(INPUT_DIR / frames[0]))
    H, W = test_img.shape[:2]
    FOV = 60 # degrees
    fx = (W / 2) / np.tan(np.deg2rad(FOV / 2))
    fy = fx
    cx = W / 2
    cy = H / 2
    intrinsics = jnp.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    tracker = PersistenceTracker(min_persistence=3)

    # Pipeline State
    prev_keypoints = None
    prev_descriptors = None
    prev_image_shape = None

    # Global Point Cloud Accumulator
    # List of (Points, Confidence) tuples
    global_point_cloud = []

    # Current global transform (World -> Camera) or (Camera -> World)?
    # We want T_world_to_current.
    # Let's track T_current_to_world.
    # Initially Identity.
    current_pose = jnp.eye(4) # [R t; 0 1]

    print(f"Starting pipeline on {len(frames)} frames...")

    # Masks
    mask_radial = radial_confidence_mask((H, W))
    mask_boundary = boundary_mask((H, W), border_percent=0.10)

    for i in tqdm(range(len(frames))):
        img_float, depth_float, name = load_image_pair(i, frames)
        if img_float is None:
            continue

        # 1. Feature Extraction (SuperPoint)
        inp = jnp.array(img_float)[None, ..., None]
        out = sp_jax(inp, training=False)

        valid_count = int(out['valid_counts'][0])
        kpts = out['keypoints'][0, :valid_count]
        scores = out['scores'][0, :valid_count]
        desc = out['descriptors'][0, :valid_count]

        # 2. Depth Processing
        depth_jnp = jnp.array(depth_float)
        mask_edge = depth_edge_mask(depth_jnp)

        # Combined Mask for later weighting
        # We need to sample these masks at keypoint locations
        # Convert kpts to integers for indexing
        kpts_int = jnp.round(kpts).astype(jnp.int32)
        # Clip to bounds
        kpts_int = jnp.clip(kpts_int, 0, jnp.array([W-1, H-1]))

        w_radial = mask_radial[kpts_int[:, 1], kpts_int[:, 0]]
        w_boundary = mask_boundary[kpts_int[:, 1], kpts_int[:, 0]]
        w_edge = mask_edge[kpts_int[:, 1], kpts_int[:, 0]]

        # Confidence Gating for Geometry
        spatial_confidence = w_radial * w_boundary * w_edge

        # 3. Matching & Tracking
        current_valid_indices = []

        # Lift *all* current points to 3D for storage/next step
        z_vals = depth_jnp[kpts_int[:, 1], kpts_int[:, 0]]
        points_3d = lift_to_3d(kpts, z_vals, intrinsics)

        if prev_keypoints is not None:
            # Match Prev -> Curr
            data = {
                'image0': {
                    'keypoints': prev_keypoints[None, ...],
                    'descriptors': prev_descriptors[None, ...],
                    'image_size': prev_image_shape
                },
                'image1': {
                    'keypoints': kpts[None, ...],
                    'descriptors': desc[None, ...],
                    'image_size': jnp.array([[W, H]])
                }
            }
            res = lg_jax(data)
            matches = np.array(res['matches0'][0]) # Indices in current image
            scores_match = np.array(res['matching_scores0'][0])

            # Update Tracker
            valid_curr_idx, valid_prev_idx = tracker.update(matches)

            # Determine Alignment Set (Tier 1: Persistent, Tier 2: Fallback)
            align_curr_idx = valid_curr_idx
            align_prev_idx = valid_prev_idx

            # Fallback if insufficient persistent points (e.g., initialization)
            if len(align_curr_idx) < 10:
                prev_indices = np.where(matches > -1)[0]
                curr_indices = matches[prev_indices]
                align_curr_idx = curr_indices
                align_prev_idx = prev_indices

            # 4. Geometry & Alignment
            if len(align_curr_idx) > 10: # Need enough points for Kabsch

                # Retrieve prev 3D points
                p3d_prev = locals().get('prev_points_3d', None)

                if p3d_prev is not None:
                    p3d_curr_subset = points_3d[align_curr_idx]
                    p3d_prev_subset = p3d_prev[align_prev_idx]

                    # Compute Weights
                    conf_curr = spatial_confidence[align_curr_idx]
                    conf_match = scores_match[align_prev_idx]
                    weights = conf_curr * conf_match

                    # Kabsch: Align Curr -> Prev
                    R, t = kabsch_weighted(p3d_curr_subset, p3d_prev_subset, weights)

                    # Reprojection Gate
                    errors = compute_reprojection_error(p3d_curr_subset, p3d_prev_subset, R, t)

                    # Update Pose
                    T_step = jnp.eye(4)
                    T_step = T_step.at[:3, :3].set(R)
                    T_step = T_step.at[:3, 3].set(t)

                    current_pose = current_pose @ T_step

                    # Accumulate Global Cloud (Strictly Persistent Points Only)
                    # Use valid_curr_idx (Persistent) for Mapping
                    if len(valid_curr_idx) > 0:
                        # Re-verify persistent points against current pose alignment?
                        # Yes, using reprojection error gate with CURRENT pose alignment

                        # Get 3D points for persistent set
                        p3d_persist_curr = points_3d[valid_curr_idx]
                        p3d_persist_prev = p3d_prev[valid_prev_idx]

                        # Check error using computed R, t
                        persist_errors = compute_reprojection_error(p3d_persist_curr, p3d_persist_prev, R, t)
                        persist_mask = persist_errors < 0.05 # 5cm gate

                        good_indices = valid_curr_idx[persist_mask]
                        if len(good_indices) > 0:
                            p3d_good = points_3d[good_indices]
                            conf_good = spatial_confidence[good_indices]

                            # Transform to World
                            p3d_good_h = jnp.pad(p3d_good, ((0, 0), (0, 1)), constant_values=1.0)
                            p3d_world_h = (current_pose @ p3d_good_h.T).T
                            p3d_world = p3d_world_h[:, :3]

                            global_point_cloud.append((p3d_world, conf_good))

        # Update State
        prev_keypoints = kpts
        prev_descriptors = desc
        prev_image_shape = jnp.array([[W, H]])
        prev_points_3d = points_3d

    # Export
    print("Exporting results...")
    if not global_point_cloud:
        print("No points generated.")
        return

    all_points = jnp.concatenate([p for p, c in global_point_cloud], axis=0)
    all_confs = jnp.concatenate([c for p, c in global_point_cloud], axis=0)

    # Save .GLB
    glb_path = OUTPUT_DIR / 'pipeline_output.glb'
    export_to_glb(np.array(all_points), np.array(all_confs), glb_path)

    # Save .LAS
    las_path = OUTPUT_DIR / 'pipeline_output.las'
    export_to_las(np.array(all_points), np.array(all_confs), las_path)

if __name__ == "__main__":
    run_pipeline()
