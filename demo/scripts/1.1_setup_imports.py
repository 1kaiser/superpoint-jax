import torch
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import subprocess
import random
from PIL import Image
import sys
from pathlib import Path

# Add the repository root to the path
if 'ipykernel' in sys.modules:
    ROOT_DIR = Path('..')
else:
    ROOT_DIR = Path(__file__).parent.parent.parent

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'LightGlue'))

from superpoint_jax.model.superpoint_torch import SuperPointTorch
from superpoint_jax.model.superpoint_jax import SuperPointJAX
from superpoint_jax.model.superglue_torch import SuperGlue as SuperGlueTorch
from superpoint_jax.model.superglue_jax import SuperGlueJAX
from superpoint_jax.model.lightglue_jax import LightGlueJAX
from superpoint_jax.utils.convert_to_jax import convert_superpoint_weights, convert_superglue_weights, convert_lightglue_weights
from flax import nnx
from lightglue import LightGlue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
