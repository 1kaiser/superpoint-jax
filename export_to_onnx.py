"""
Export SuperPoint PyTorch model to ONNX format for browser inference via jax-js.

Usage:
    conda run -n num_python python export_to_onnx.py

Outputs:
    superpoint.onnx  (~5MB)
"""
import sys
sys.path.insert(0, '.')

import torch
import os

from superpoint_jax.model.superpoint_torch import SuperPointTorch


def main():
    print("=" * 50)
    print("SuperPoint PyTorch → ONNX Export")
    print("=" * 50)

    # Load model
    print("\n1. Loading PyTorch model...")
    model = SuperPointTorch()
    weights = torch.load('weights/superpoint_torch.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    print("   Model loaded ✓")

    # Test inference
    print("\n2. Testing inference...")
    dummy = torch.randn(1, 1, 480, 640)
    with torch.no_grad():
        # Get intermediate outputs for ONNX export
        features = model.backbone(dummy)
        scores = model.detector(features)
        scores = torch.nn.functional.softmax(scores, dim=1)[:, :-1]
        desc = model.descriptor(features)
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)
    print(f"   Scores shape: {scores.shape}")
    print(f"   Descriptors shape: {desc.shape}")

    # Export to ONNX - we export the backbone+heads as a single model
    print("\n3. Exporting to ONNX...")

    class SuperPointONNX(torch.nn.Module):
        """Wrapper that outputs raw scores and descriptors for post-processing in JS."""
        def __init__(self, model):
            super().__init__()
            self.backbone = model.backbone
            self.detector = model.detector
            self.descriptor = model.descriptor

        def forward(self, image):
            features = self.backbone(image)

            # Detector: softmax scores (drop dustbin)
            scores = self.detector(features)
            scores = torch.nn.functional.softmax(scores, dim=1)[:, :-1]

            # Descriptor: L2 normalized
            desc = self.descriptor(features)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)

            return scores, desc

    onnx_model = SuperPointONNX(model)
    onnx_model.eval()

    onnx_path = 'superpoint.onnx'
    torch.onnx.export(
        onnx_model,
        dummy,
        onnx_path,
        input_names=['image'],
        output_names=['scores', 'descriptors'],
        dynamic_axes={
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'scores': {0: 'batch', 2: 'h8', 3: 'w8'},
            'descriptors': {0: 'batch', 2: 'h8', 3: 'w8'},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx as onnx_lib
    onnx_loaded = onnx_lib.load(onnx_path)
    onnx_lib.checker.check_model(onnx_loaded)

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n✅ ONNX model saved: {onnx_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Validation passed ✓")

    # Verify with ONNX Runtime
    print("\n4. Verifying with ONNX Runtime...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 1, 480, 640).astype(np.float32)
    out_scores, out_desc = session.run(None, {'image': test_input})
    print(f"   ORT scores shape: {out_scores.shape}")
    print(f"   ORT descriptors shape: {out_desc.shape}")
    print(f"   ✓ ONNX Runtime inference successful")


if __name__ == '__main__':
    main()
