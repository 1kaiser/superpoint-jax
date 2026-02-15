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

A JavaScript version of the SuperPoint inference pipeline is available in `jax-js/`, using the ONNX model exported from the PyTorch weights.

### Quick Start

```bash
# 1. Export PyTorch model to ONNX (requires num_python conda env)
conda run -n num_python python export_to_onnx.py

# 2. Install JS dependencies
cd jax-js && npm install

# 3. Run inference
node superpoint_jax_js.js --test              # synthetic test pattern
node superpoint_jax_js.js path/to/image.jpg   # real image
```

### JS Pipeline

The `superpoint_jax_js.js` script implements the full SuperPoint pipeline:
1. **Image loading** — grayscale, normalize [0,1], pad to stride-8
2. **ONNX inference** — backbone + detector + descriptor heads (~200ms on CPU)
3. **Pixel shuffle** — (64, H/8, W/8) → (H, W) score map
4. **NMS** — non-maximum suppression with configurable radius
5. **Keypoint extraction** — thresholding + top-k selection
6. **Descriptor sampling** — bilinear interpolation + L2 normalization
