# SuperPoint Inference with JAX/Flax

This repository provides an inference-only pipeline for keypoint detection using the SuperPoint model. It includes a PyTorch implementation of SuperPoint, a JAX/Flax (NNX) implementation, and a conversion script to transfer pretrained weights from PyTorch to JAX. Inference can then be performed on input images using the converted JAX model.

## Overview

- **PyTorch Model:** `superpoint_torch.py` implements the SuperPoint model in PyTorch.
- **JAX Model:** `superpoint_jax.py` contains the corresponding SuperPoint model implementation in JAX/Flax NNX.
- **Weight Conversion:** `convert_to_jax.py` includes helper functions to copy convolution and batch normalization parameters from the PyTorch model to the JAX model.
- **Demo:** A Jupyter Notebook (`demo/compare_jax_torch.ipynb`) demonstrates the entire process—from converting weights to running inference and visualizing keypoints.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [JAX](https://github.com/google/jax) and [jaxlib](https://github.com/google/jax)
- [Flax](https://flax.readthedocs.io/)
- NumPy
- Matplotlib
- OpenCV (cv2)

## Usage

1. **Obtain Pretrained Weights:**  
   Place the pretrained PyTorch SuperPoint weights (e.g., `superpoint_torch_weights.pth`) in the repository root or a designated folder.

2. **Convert Weights:**  
   Run the conversion script to create a JAX model with copied weights and save the converted state:
   ```bash
   python convert_superpoint_model.py

## JavaScript Inference (ONNX Runtime)

A standalone Node.js inference pipeline is available in `jax-js/`, providing high-performance SuperPoint detection (~200ms on CPU) without JAX or PyTorch.

### Quick Start

```bash
# 1. Export PyTorch model to ONNX
python export_to_onnx.py

# 2. Install JS dependencies (ONNX Runtime + Sharp)
cd jax-js && npm install

# 3. Run inference
node superpoint_jax_js.js --test              # Synthetic test pattern
node superpoint_jax_js.js path/to/image.jpg   # Real image
```

### 3-Way Notebook Comparison

The `demo/compare_jax_torch.ipynb` notebook has been updated to include a **triple comparison**:
1. **PyTorch** (Ground Truth)
2. **JAX/Flax (NNX)** (Weight-converted implementation)
3. **JS/ONNX** (Node.js standalone pipeline)

The notebook uses `subprocess` to run the JS pipeline and visualizes keypoint consistency across all three backends.

### Validation Results (JS vs PyTorch)

The JS pipeline has been meticulously tuned to match `cv2` preprocessing (grayscale coefficients and zero-padding). 

| Dataset | Match Rate (±2px) | Notes |
|---------|-------------------|-------|
| Sacré-Cœur | **100.0%** | Grayscale input |
| Synthetic Patterns | **100.0%** | Checker, Noise, Shapes |
| Real RGB Frames | **~93-95%** | Minor rounding diffs in RGB->Gray conversion |

### Pipeline Details

The `superpoint_jax_js.js` script implements:
- **CV2-Compatible Preprocessing**: Manual grayscale weighting `round(0.299R + 0.587G + 0.114B)` and zero-padding.
- **Efficient NMS**: Non-maximum suppression with configurable radius (default 4).
- **Descriptor Sampling**: High-fidelity bilinear interpolation matching the original PyTorch implementation.
