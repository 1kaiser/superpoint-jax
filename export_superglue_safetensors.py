#!/usr/bin/env python3
"""Export SuperGlue PyTorch pretrained weights to safetensors format.

Downloads the pretrained weights from the magicleap/SuperGluePretrainedNetwork
GitHub repo and converts them to safetensors format for use with @jax-js/loaders.

Usage:
    conda run -n num_python python export_superglue_safetensors.py
"""

import os
import torch
import numpy as np
from safetensors.torch import save_file
from collections import OrderedDict

# URLs for pretrained weights (from magicleap/SuperGluePretrainedNetwork)
WEIGHT_URLS = {
    "indoor": "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_indoor.pth",
    "outdoor": "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_outdoor.pth",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights")
os.makedirs(OUTPUT_DIR, exist_ok=True)



def download_weights(name, url):
    """Download weights if not already cached."""
    cache_path = os.path.join(OUTPUT_DIR, f"superglue_{name}.pth")
    if os.path.exists(cache_path):
        print(f"  Using cached: {cache_path}")
        return cache_path

    print(f"  Downloading from {url}...")
    torch.hub.download_url_to_file(url, cache_path)
    return cache_path


def export_to_safetensors(name, pth_path):
    """Convert a SuperGlue .pth state_dict to safetensors."""
    print(f"\n{'='*60}")
    print(f"  Exporting {name} weights")
    print(f"{'='*60}")

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # Print architecture summary
    print(f"\n  Total tensors: {len(state_dict)}")
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"  Total parameters: {total_params:,}")

    # Group by component
    components = {}
    for key in state_dict:
        component = key.split(".")[0]
        components.setdefault(component, []).append(key)

    print(f"\n  Components:")
    for comp, keys in components.items():
        params = sum(state_dict[k].numel() for k in keys)
        print(f"    {comp}: {len(keys)} tensors, {params:,} params")

    # Print shape info for each tensor
    print(f"\n  Tensor details:")
    for key, tensor in state_dict.items():
        print(f"    {key}: {list(tensor.shape)} ({tensor.dtype})")

    # Save as safetensors (keys are preserved as-is for WeightMapper compatibility)
    # safetensors requires contiguous float32 tensors
    clean_dict = OrderedDict()
    for key, tensor in state_dict.items():
        clean_dict[key] = tensor.contiguous().float()

    out_path = os.path.join(OUTPUT_DIR, f"superglue_{name}.safetensors")
    save_file(clean_dict, out_path)
    file_size = os.path.getsize(out_path)
    print(f"\n  ✓ Saved: {out_path} ({file_size / 1024 / 1024:.1f} MB)")

    return out_path


def main():
    print("SuperGlue Weight Export: PyTorch .pth → safetensors")
    print("=" * 60)

    for name, url in WEIGHT_URLS.items():
        print(f"\n📦 {name} weights:")
        pth_path = download_weights(name, url)
        export_to_safetensors(name, pth_path)

    print(f"\n{'='*60}")
    print("✓ All exports complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
