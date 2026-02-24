import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np
import cv2
import os
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

# Local imports
from lightglue_jax.models.superpoint import SuperPoint
from lightglue_jax.models.lightglue import LightGlue

def load_image(path, target_size=(1024, 1024)):
    img = cv2.imread(path)
    if img is None:
        return None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_resized = cv2.resize(img_gray, target_size)
    input_tensor = jnp.array(img_gray_resized[None, ..., None] / 255.0)
    return img_gray_resized, input_tensor

def get_top_k(sp_out, k=1024, threshold=0.005):
    scores = sp_out['scores'][0]
    desc = sp_out['descriptors'][0]
    indices = jnp.where(scores > threshold)
    y, x = indices
    valid_scores = scores[indices]
    top_k_indices = jnp.argsort(valid_scores)[::-1][:k]
    y, x = y[top_k_indices], x[top_k_indices]
    kpts = jnp.stack([x, y], axis=-1).astype(jnp.float32)
    iy = jnp.clip((y / 8).astype(jnp.int32), 0, desc.shape[0]-1)
    ix = jnp.clip((x / 8).astype(jnp.int32), 0, desc.shape[1]-1)
    return kpts, desc[iy, ix, :], valid_scores[top_k_indices]

def run_analysis():
    # 1. Setup Models
    sp_model = SuperPoint()
    lg_model = LightGlue(n_layers=9)
    
    with open("weights/superpoint.msgpack", "rb") as f:
        sp_vars = serialization.from_bytes(None, f.read())
    with open("weights/superpoint_lightglue.msgpack", "rb") as f:
        lg_vars = serialization.from_bytes(None, f.read())
        
    jit_sp = jax.jit(sp_model.apply)
    jit_lg = jax.jit(lg_model.apply)
    
    base_frame_idx = 0
    gaps = [1, 50, 100, 200]
    results = []
    
    img_dir = "data/input_frames"
    base_path = os.path.join(img_dir, f"frame_{base_frame_idx:04d}.png")
    _, input0 = load_image(base_path)
    print(f"Pre-extracting features for base frame: {base_path}")
    sp0 = jit_sp(sp_vars, input0)
    kpts0, desc0, _ = get_top_k(sp0)
    
    # Warmup
    _ = jit_lg(lg_vars, {"image0": {"keypoints": kpts0[None], "descriptors": desc0[None]}, "image1": {"keypoints": kpts0[None], "descriptors": desc0[None]}})

    for gap in gaps:
        target_idx = base_frame_idx + gap
        target_path = os.path.join(img_dir, f"frame_{target_idx:04d}.png")
        
        _, input1 = load_image(target_path)
        if input1 is None:
            print(f"Skipping gap {gap}, frame {target_path} not found.")
            continue
            
        # Extract
        sp1 = jit_sp(sp_vars, input1)
        kpts1, desc1, _ = get_top_k(sp1)
        
        # Match
        data = {
            "image0": {"keypoints": kpts0[None], "descriptors": desc0[None]},
            "image1": {"keypoints": kpts1[None], "descriptors": desc1[None]}
        }
        
        start_time = time.time()
        lg_out = jit_lg(lg_vars, data)
        match_time = (time.time() - start_time) * 1000
        
        # Process matches
        log_scores = lg_out['scores'][0, :-1, :-1]
        m0 = jnp.argmax(log_scores, axis=1)
        m1 = jnp.argmax(log_scores, axis=0)
        mutual = (jnp.arange(len(m0)) == m1[m0])
        conf = jnp.exp(jnp.max(log_scores, axis=1))
        valid = mutual & (conf > 0.1)
        
        num_matches = int(jnp.sum(valid))
        avg_conf = float(jnp.mean(conf[valid])) if num_matches > 0 else 0
        
        print(f"Gap {gap:3d}: Matches={num_matches:4d}, Avg Conf={avg_conf:.4f}, Time={match_time:.2f}ms")
        results.append([gap, num_matches, f"{avg_conf:.4f}", f"{match_time:.2f}ms"])
        
        # Save visualization for the gap
        idx0 = jnp.where(valid)[0]
        idx1 = m0[idx0]
        
        h0, w0 = input0.shape[1:3]
        h1, w1 = input1.shape[1:3]
        
        # Load RGB for visualization if available
        rgb0 = cv2.imread(base_path)
        rgb0 = cv2.resize(cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB), (w0, h0))
        rgb1 = cv2.imread(target_path)
        rgb1 = cv2.resize(cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB), (w1, h1))
        
        combined_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
        combined_img[:h0, :w0, :] = rgb0
        combined_img[:h1, w0:, :] = rgb1
        
        plt.figure(figsize=(20, 10))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title(f"LightGlue JAX Matches - Gap {gap} ({num_matches} matches)")
        
        matched_kpts0 = kpts0[idx0]
        matched_kpts1 = kpts1[idx1]
        
        for p0, p1 in zip(matched_kpts0, matched_kpts1):
            plt.plot([p0[0], p1[0] + w0], [p0[1], p1[1]], 'c-', linewidth=0.5, alpha=0.5)
            
        vis_path = f"output/matches_gap_{gap:03d}.png"
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    print("\nExtended Frame Gap Analysis Results:")
    print(tabulate(results, headers=["Gap", "Matches", "Avg Conf", "Inference Time"], tablefmt="github"))

if __name__ == "__main__":
    run_analysis()
