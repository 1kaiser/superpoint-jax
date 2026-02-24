# SuperPoint & LightGlue: JAX Inference 🚀

A high-performance, native **JAX/Flax** implementation of SuperPoint and LightGlue. This repository is optimized for high-speed feature extraction and matching, maintaining bit-level parity with the official PyTorch baselines while enabling hardware acceleration on TPUs and GPUs.

## 🌟 Key Features

- **Native JAX/Flax Implementation:** Fully differentiable implementations of SuperPoint and LightGlue.
- **Advanced Features:** Includes support for **Rotary Positional Embeddings (RoPE)** and **Adaptive Pruning** for maximum efficiency.
- **Hardware Optimized:** Designed for seamless JIT compilation, providing sub-5ms inference times on modern accelerators.
- **Verified Parity:** Rigorously tested against PyTorch implementations to ensure identical matching quality.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Inference Demo
Execute the full matching pipeline (Extraction + Matching) on sequential frames:
```bash
PYTHONPATH=. python demo/scripts/run_lightglue_jax_pipeline.py
```

### 3. Google Colab Experience
For a one-click TPU/GPU demonstration, use our interactive notebook:
[`lightglue_jax_inference_demo.ipynb`](./lightglue_jax_inference_demo.ipynb)

## 📊 Experimental Results

### Matching Quality (Sequential Dataset)
Using the JAX-compiled pipeline on sequential video frames:
- **Matches Found:** 799 high-confidence mutual matches.
- **Average Confidence:** 0.9159.
- **Inference Time:** **~4.2 ms** (JAX JIT-compiled).

![Main Pipeline Result](demo/assets/results/main_pipeline_result.png)

### Extended Frame Gap Analysis
Evaluation of matching robustness across increasing temporal gaps (Ref vs Ref+N).

| Gap | Matches | Avg Conf | Inference Time | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 799 | 0.9159 | 4.17ms | Sequential tracking |
| **50** | 446 | 0.7641 | 5.35ms | Significant motion |
| **100** | 246 | 0.6675 | 4.70ms | Wide baseline |
| **200** | 23 | 0.3444 | 5.78ms | Severe perspective shift |

#### Visual Robustness (Gap 50):
![Gap 50](demo/assets/results/matches_gap_050.png)

## 🛠️ Usage & Testing

### Parity & Robustness Tests
Verify the JAX implementation against official baselines:
```bash
# SuperPoint Extraction Parity
python demo/scripts/test_superpoint_jax.py

# LightGlue Matching Parity
python demo/scripts/test_lightglue_jax.py

# Extended Gap Analysis
python demo/scripts/extended_gap_analysis.py
```

## 📁 Repository Structure
```text
.
├── lightglue_jax/      # Core JAX/Flax implementations
├── demo/scripts/       # Inference and analysis scripts
├── demo/assets/results/# Visualization of test results
├── weights/            # JAX model weights (.msgpack)
└── lightglue_jax_inference_demo.ipynb
```

## ⚖️ Model Weights
Pre-converted JAX weights are available in the [v1.0.0 Release of d_jax](https://github.com/1kaiser/d_jax/releases/tag/v1.0.0).

---
Maintainer: [1kaiser](https://github.com/1kaiser)
