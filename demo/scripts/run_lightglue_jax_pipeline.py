import jax
import jax.numpy as jnp
from flax import serialization
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
from lightglue_jax.models.superpoint import SuperPoint
from lightglue_jax.models.lightglue import LightGlue

def load_image(path, target_size=(1024, 1024)):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize for inference
    img_gray_resized = cv2.resize(img_gray, target_size)
    input_tensor = jnp.array(img_gray_resized[None, ..., None] / 255.0)
    
    return img_rgb, input_tensor, img_gray_resized

def run_pipeline():
    # 1. Setup Models
    sp_model = SuperPoint()
    lg_model = LightGlue(n_layers=9)
    
    # 2. Load Weights
    sp_weights_path = "weights/superpoint.msgpack"
    lg_weights_path = "weights/superpoint_lightglue.msgpack"
    
    with open(sp_weights_path, "rb") as f:
        sp_vars = serialization.from_bytes(None, f.read())
    with open(lg_weights_path, "rb") as f:
        lg_vars = serialization.from_bytes(None, f.read())
        
    # JIT Compile
    jit_sp = jax.jit(sp_model.apply)
    jit_lg = jax.jit(lg_model.apply)
    
    # 3. Load Images
    img0_path = "demo/assets/sacre_coeur1.jpg"
    img1_path = "demo/assets/sacre_coeur2.jpg"
    
    rgb0, input0, gray0 = load_image(img0_path)
    rgb1, input1, gray1 = load_image(img1_path)
    
    # 4. Extract SuperPoint Features
    print("Extracting SuperPoint features (JAX)...")
    sp0 = jit_sp(sp_vars, input0)
    sp1 = jit_sp(sp_vars, input1)
    
    # 5. Prepare data for LightGlue
    # Scores and Descriptors are [1, N] and [1, N, 256]
    # We select top K keypoints
    k = 1024
    
    def get_top_k(sp_out, k, threshold=0.005):
        scores = sp_out['scores'][0]
        desc = sp_out['descriptors'][0]
        
        # 1. Filter by threshold
        indices = jnp.where(scores > threshold)
        y, x = indices
        valid_scores = scores[indices]
        
        # 2. Get top K
        top_k_indices = jnp.argsort(valid_scores)[::-1][:k]
        y, x = y[top_k_indices], x[top_k_indices]
        
        kpts = jnp.stack([x, y], axis=-1).astype(jnp.float32)
        
        # 3. Sample descriptors (bilinear or nearest)
        # descriptors: [1, H/8, W/8, 256]
        # Coordinates in [0, H-1]
        iy = jnp.clip((y / 8).astype(jnp.int32), 0, desc.shape[0]-1)
        ix = jnp.clip((x / 8).astype(jnp.int32), 0, desc.shape[1]-1)
        kpts_desc = desc[iy, ix, :]
        
        return kpts, kpts_desc, valid_scores[top_k_indices]

    kpts0, desc0, scores0 = get_top_k(sp0, k)
    kpts1, desc1, scores1 = get_top_k(sp1, k)
    
    data = {
        "image0": {"keypoints": kpts0[None], "descriptors": desc0[None]},
        "image1": {"keypoints": kpts1[None], "descriptors": desc1[None]}
    }
    
    # 6. Run LightGlue Matching
    print("Matching with LightGlue (JAX)...")
    lg_out = jit_lg(lg_vars, data)
    
    # 7. Visualize Matches
    # scores: [B, M+1, N+1]
    log_scores = lg_out['scores'][0]
    
    # Mutual nearest neighbors
    # Remove dustbins
    scores = log_scores[:-1, :-1]
    
    m0 = jnp.argmax(scores, axis=1)
    m1 = jnp.argmax(scores, axis=0)
    mutual = (jnp.arange(len(m0)) == m1[m0])
    
    # Confidence threshold (exp because they are log-scores)
    # The scores already incorporate matchability
    # LightGlue usually uses 0.1 on the assignment probability
    conf = jnp.exp(jnp.max(scores, axis=1))
    valid = mutual & (conf > 0.1)
    
    idx0 = jnp.where(valid)[0]
    idx1 = m0[idx0]
    
    matched_kpts0 = kpts0[idx0]
    matched_kpts1 = kpts1[idx1]
    
    print(f"Found {len(matched_kpts0)} matches.")
    
    # Combine images side by side
    h0, w0 = gray0.shape
    h1, w1 = gray1.shape
    combined_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    combined_img[:h0, :w0, :] = cv2.resize(rgb0, (w0, h0))
    combined_img[:h1, w0:, :] = cv2.resize(rgb1, (w1, h1))
    
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_img)
    plt.axis('off')
    
    for p0, p1 in zip(matched_kpts0, matched_kpts1):
        plt.plot([p0[0], p1[0] + w0], [p0[1], p1[1]], 'c-', linewidth=0.5)
        plt.scatter([p0[0], p1[0] + w0], [p0[1], p1[1]], c='g', s=1)
        
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/lightglue_jax_matches.png")
    print("Saved visualization to output/lightglue_jax_matches.png")

if __name__ == "__main__":
    run_pipeline()
