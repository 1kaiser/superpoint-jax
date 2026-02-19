import torch
import jax
import jax.numpy as jnp
import numpy as np
import cv2
import os
import time
import random
import sys
from pathlib import Path
from pandas import DataFrame
from tabulate import tabulate

# Add the repository root to the path
ROOT_DIR = Path('/app')
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'LightGlue'))

from superpoint_jax.model.superpoint_torch import SuperPointTorch
from superpoint_jax.model.superpoint_jax import SuperPointJAX
from superpoint_jax.model.superglue_jax import SuperGlueJAX
from superpoint_jax.model.lightglue_jax import LightGlueJAX
from superpoint_jax.utils.convert_to_jax import convert_superpoint_weights, convert_superglue_weights, convert_lightglue_weights
from flax import nnx
from lightglue import LightGlue
from superpoint_jax.model.superglue_torch import SuperGlue as SuperGlueTorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Models
print("Loading models...")
sp_torch = SuperPointTorch(max_num_keypoints=1024).to(device)
sp_torch.load_state_dict(torch.load(str(ROOT_DIR / 'weights/superpoint_torch.pth'), map_location=device))
sp_torch.eval()

sp_jax = SuperPointJAX(max_num_keypoints=1024, rngs=nnx.Rngs(0))
sp_jax = convert_superpoint_weights(sp_torch, sp_jax)

lg_torch = LightGlue(features='superpoint').to(device)
lg_torch.eval()

lg_jax = LightGlueJAX(rngs=nnx.Rngs(0))
lg_jax = convert_lightglue_weights(lg_torch, lg_jax)

sg_torch = SuperGlueTorch({'weights': 'indoor'}).to(device)
sg_torch.eval()

sg_jax = SuperGlueJAX(rngs=nnx.Rngs(0))
sg_jax = convert_superglue_weights(sg_torch, sg_jax)

print("All models loaded.")

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img_float = img.astype(np.float32) / 255.0
    return img, img_float

def run_pytorch_lg(img0_float, img1_float):
    with torch.no_grad():
        inp0 = torch.from_numpy(img0_float).unsqueeze(0).unsqueeze(0).to(device)
        inp1 = torch.from_numpy(img1_float).unsqueeze(0).unsqueeze(0).to(device)
        out0 = sp_torch({'image': inp0})
        out1 = sp_torch({'image': inp1})
        data = {
            'image0': {'keypoints': out0['keypoints'][0].unsqueeze(0), 'descriptors': out0['descriptors'][0].unsqueeze(0), 'image_size': torch.tensor([[inp0.shape[3], inp0.shape[2]]], device=device).float()},
            'image1': {'keypoints': out1['keypoints'][0].unsqueeze(0), 'descriptors': out1['descriptors'][0].unsqueeze(0), 'image_size': torch.tensor([[inp1.shape[3], inp1.shape[2]]], device=device).float()}
        }
        res = lg_torch(data)
        return {'matches0': res['matches0'][0].cpu().numpy(), 'scores0': res['matching_scores0'][0].cpu().numpy()}

def run_jax_lg(img0_float, img1_float):
    inp0 = jnp.array(img0_float)[None, ..., None]
    inp1 = jnp.array(img1_float)[None, ..., None]
    out0 = sp_jax(inp0, training=False)
    out1 = sp_jax(inp1, training=False)
    v0, v1 = int(out0['valid_counts'][0]), int(out1['valid_counts'][0])
    data = {
        'image0': {'keypoints': out0['keypoints'][:, :v0], 'descriptors': out0['descriptors'][:, :v0], 'image_size': jnp.array([[img0_float.shape[1], img0_float.shape[0]]])},
        'image1': {'keypoints': out1['keypoints'][:, :v1], 'descriptors': out1['descriptors'][:, :v1], 'image_size': jnp.array([[img1_float.shape[1], img1_float.shape[0]]])}
    }
    res = lg_jax(data)
    return {'matches0': np.array(res['matches0'][0]), 'scores0': np.array(res['matching_scores0'][0])}

def run_jax_sg(img0_float, img1_float):
    inp0 = jnp.array(img0_float)[None, ..., None]
    inp1 = jnp.array(img1_float)[None, ..., None]
    out0 = sp_jax(inp0, training=False)
    out1 = sp_jax(inp1, training=False)
    v0, v1 = int(out0['valid_counts'][0]), int(out1['valid_counts'][0])
    data = {
        'keypoints0': out0['keypoints'][:, :v0], 'scores0': out0['scores'][:, :v0], 'descriptors0': out0['descriptors'][:, :v0].transpose(0, 2, 1), 'image0_shape': (1, 1, *img0_float.shape),
        'keypoints1': out1['keypoints'][:, :v1], 'scores1': out1['scores'][:, :v1], 'descriptors1': out1['descriptors'][:, :v1].transpose(0, 2, 1), 'image1_shape': (1, 1, *img1_float.shape),
    }
    res = sg_jax(data, training=False)
    return {'matches0': np.array(res['matches0'][0]), 'scores0': np.array(res['matching_scores0'][0])}

# Benchmark Loop
dataset_path = ROOT_DIR / 'demo/frames/input_frames'
frames = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])

NUM_PAIRS = 5
GAP = 10
print(f"Starting benchmark on {NUM_PAIRS} random pairs with gap {GAP}...")

implementations = [
    ("LightGlue PyTorch", run_pytorch_lg),
    ("LightGlue JAX", run_jax_lg),
    ("SuperGlue JAX", run_jax_sg)
]

stats = {name: {'matches': [], 'time': []} for name, _ in implementations}

# Warmup JAX
print("Warming up JAX models...")
dummy_img = np.zeros((480, 640), dtype=np.float32)
run_jax_lg(dummy_img, dummy_img)
run_jax_sg(dummy_img, dummy_img)

for i in range(NUM_PAIRS):
    idx0 = random.randint(0, len(frames) - GAP - 1)
    idx1 = idx0 + GAP
    print(f"Pair {i+1}: {frames[idx0]} vs {frames[idx1]}")

    _, img0 = load_image(dataset_path / frames[idx0])
    _, img1 = load_image(dataset_path / frames[idx1])

    for name, run_func in implementations:
        start = time.time()
        res = run_func(img0, img1)
        end = time.time()

        num_matches = np.sum(res['matches0'] > -1)
        stats[name]['matches'].append(num_matches)
        stats[name]['time'].append((end - start) * 1000)

# Aggregate
final_results = []
for name, _ in implementations:
    m = stats[name]['matches']
    t = stats[name]['time']
    final_results.append({
        'Implementation': name,
        'Avg Matches': f"{np.mean(m):.1f}",
        'Avg Time (ms)': f"{np.mean(t):.2f}",
        'Std Time (ms)': f"{np.std(t):.2f}",
        'Min Time (ms)': f"{np.min(t):.2f}",
        'Max Time (ms)': f"{np.max(t):.2f}"
    })

df = DataFrame(final_results)
print("\nFinal Benchmark Results (Real Data, Gap 10):")
print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

# Update 1.5_comparison_table.py with this logic for the notebook
with open(ROOT_DIR / 'demo/scripts/1.5_comparison_table.py', 'w') as f:
    f.write(f"""
# Auto-generated from benchmark_real_data.py
import time
from pandas import DataFrame
from tabulate import tabulate

results = {final_results}
df = DataFrame(results)
print("\\nComparison Table (Averaged over random pairs):")
print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
""")
