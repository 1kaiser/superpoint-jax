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

def load_image(path, max_size=1024):
    img = cv2.imread(path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    
    # Maintain aspect ratio
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Round to nearest multiple of 8 for SuperPoint
    new_h = (new_h // 8) * 8
    new_w = (new_w // 8) * 8
    
    img_resized = cv2.resize(img, (new_w, new_h))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    input_tensor = jnp.array(img_gray[None, ..., None] / 255.0)
    return img_resized, input_tensor, (h, w)

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
    gaps = [1] + list(range(5, 201, 5))
    results = []
    
    img_dir = "data/input_frames"
    base_path = os.path.join(img_dir, f"frame_{base_frame_idx:04d}.png")
    img0_resized, input0, _ = load_image(base_path)
    print(f"Pre-extracting features for base frame: {base_path} (Size: {img0_resized.shape[:2]})")
    sp0 = jit_sp(sp_vars, input0)
    kpts0, desc0, _ = get_top_k(sp0)
    
    # Warmup
    _ = jit_lg(lg_vars, {"image0": {"keypoints": kpts0[None], "descriptors": desc0[None]}, "image1": {"keypoints": kpts0[None], "descriptors": desc0[None]}})

    for gap in gaps:
        target_idx = base_frame_idx + gap
        target_path = os.path.join(img_dir, f"frame_{target_idx:04d}.png")
        
        img1_resized, input1, _ = load_image(target_path)
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
        
        h0, w0 = img0_resized.shape[:2]
        h1, w1 = img1_resized.shape[:2]
        
        rgb0 = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB)
        rgb1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
        
        combined_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
        combined_img[:h0, :w0, :] = rgb0
        combined_img[:h1, w0:, :] = rgb1
        
        plt.figure(figsize=(20, 10))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title(f"LightGlue JAX Matches - Gap {gap} ({num_matches} matches)")
        
        matched_kpts0 = kpts0[idx0]
        matched_kpts1 = kpts1[idx1]
        
        # Use a rainbow colormap based on x-coordinate of matched_kpts0
        cmap = plt.get_cmap('gist_rainbow')
        x0 = matched_kpts0[:, 0]
        x0_min, x0_max = x0.min(), x0.max()
        x0_norm = (x0 - x0_min) / (x0_max - x0_min + 1e-6)
        vis_colors = cmap(x0_norm)
        
        for i, (p0, p1) in enumerate(zip(matched_kpts0, matched_kpts1)):
            plt.plot([p0[0], p1[0] + w0], [p0[1], p1[1]], color=vis_colors[i], linewidth=0.75, alpha=0.6)
            
        vis_path = f"output/matches_gap_{gap:03d}.png"
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    print("\nExtended Frame Gap Analysis Results:")
    print(tabulate(results, headers=["Gap", "Matches", "Avg Conf", "Inference Time"], tablefmt="github"))

if __name__ == "__main__":
    run_analysis()
