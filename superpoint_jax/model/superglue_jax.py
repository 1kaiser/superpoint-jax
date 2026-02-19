import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Dict, Any
from functools import partial

class MLP(nnx.Module):
    """Multi-layer perceptron in Flax NNX."""
    def __init__(self, in_channels: int, channels: List[int], do_bn: bool = True, *, rngs: nnx.Rngs):
        self.layers = nnx.List([])
        curr_channels = in_channels
        for i, out_channels in enumerate(channels):
            self.layers.append(nnx.Linear(curr_channels, out_channels, use_bias=True, rngs=rngs))
            if i < len(channels) - 1:
                if do_bn:
                    self.layers.append(nnx.BatchNorm(out_channels, rngs=rngs))
                self.layers.append(nnx.relu)
            curr_channels = out_channels

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.BatchNorm):
                if training:
                    layer.train()
                else:
                    layer.eval()
                x = layer(x)
            elif callable(layer):
                x = layer(x)
        return x

def normalize_keypoints(kpts: jnp.ndarray, image_shape: Tuple[int, int, int, int]) -> jnp.ndarray:
    """Normalize keypoints locations based on image shape."""
    _, _, height, width = image_shape
    size = jnp.array([width, height], dtype=kpts.dtype)
    center = size / 2
    scaling = jnp.max(size) * 0.7
    return (kpts - center) / scaling

class KeypointEncoder(nnx.Module):
    """Joint encoding of visual appearance and location using MLPs."""
    def __init__(self, feature_dim: int, layers: List[int], *, rngs: nnx.Rngs):
        self.encoder = MLP(3, layers + [feature_dim], rngs=rngs)

    def __call__(self, kpts: jnp.ndarray, scores: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # kpts: [B, N, 2], scores: [B, N]
        inputs = jnp.concatenate([kpts, scores[:, :, None]], axis=2)
        x = self.encoder(inputs, training=training) # [B, N, feature_dim]
        return x.transpose(0, 2, 1) # [B, feature_dim, N]

def attention(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # query, key, value: [B, num_heads, N, head_dim]
    dim = query.shape[-1]
    scores = jnp.einsum('bhn d, bhm d -> bhnm', query, key) / jnp.sqrt(dim)
    prob = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('bhnm, bhmd -> bhnd', prob, value), prob

class MultiHeadedAttention(nnx.Module):
    """Multi-head attention to increase model expressivity."""
    def __init__(self, num_heads: int, d_model: int, *, rngs: nnx.Rngs):
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.proj = nnx.List([nnx.Linear(d_model, d_model, use_bias=True, rngs=rngs) for _ in range(3)])
        self.merge = nnx.Linear(d_model, d_model, use_bias=True, rngs=rngs)

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # query, key, value: [B, d_model, N]
        B, D, N = query.shape
        B, D, M = key.shape

        # Linear expects [B, N, D]. Torch Conv1d output [B, D, N]
        # Torch view(B, head_dim, num_heads, N)
        q = self.proj[0](query.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(B, self.head_dim, self.num_heads, N)
        k = self.proj[1](key.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(B, self.head_dim, self.num_heads, M)
        v = self.proj[2](value.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(B, self.head_dim, self.num_heads, M)

        # Transpose to [B, num_heads, N, head_dim] for attention
        q = q.transpose(0, 2, 3, 1)
        k = k.transpose(0, 2, 3, 1)
        v = v.transpose(0, 2, 3, 1)

        x, prob = attention(q, k, v) # [B, H, N, head_dim]
        # Transpose back to [B, head_dim, num_heads, N] to match Torch view(B, D, N)
        x = x.transpose(0, 3, 1, 2).reshape(B, D, N)
        res = self.merge(x.transpose(0, 2, 1)).transpose(0, 2, 1) # [B, D, N]
        return res, prob

class AttentionalPropagation(nnx.Module):
    def __init__(self, feature_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.attn = MultiHeadedAttention(num_heads, feature_dim, rngs=rngs)
        self.mlp = MLP(feature_dim * 2, [feature_dim * 2, feature_dim], rngs=rngs)

    def __call__(self, x: jnp.ndarray, source: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        message, prob = self.attn(x, source, source)
        inputs = jnp.concatenate([x, message], axis=1) # [B, D*2, N]
        res = self.mlp(inputs.transpose(0, 2, 1), training=training).transpose(0, 2, 1) # [B, D, N]
        return res, prob

class AttentionalGNN(nnx.Module):
    def __init__(self, feature_dim: int, layer_names: List[str], *, rngs: nnx.Rngs):
        self.layers = nnx.List([AttentionalPropagation(feature_dim, 4, rngs=rngs) for _ in range(len(layer_names))])
        self.names = layer_names

    def __call__(self, desc0: jnp.ndarray, desc1: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, List[Dict[str, Any]]]:
        all_attentions = []
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            (delta0, prob0) = layer(desc0, src0, training=training)
            (delta1, prob1) = layer(desc1, src1, training=training)
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1
            all_attentions.append({
                'name': name,
                'prob0': prob0,
                'prob1': prob1
            })
        return desc0, desc1, all_attentions

def log_sinkhorn_iterations(Z: jnp.ndarray, log_mu: jnp.ndarray, log_nu: jnp.ndarray, iters: int) -> jnp.ndarray:
    """Perform Sinkhorn Normalization in Log-space for stability."""
    u = jnp.zeros_like(log_mu)
    v = jnp.zeros_like(log_nu)

    for _ in range(iters):
        u = log_mu - jax.scipy.special.logsumexp(Z + v[:, None, :], axis=2)
        v = log_nu - jax.scipy.special.logsumexp(Z + u[:, :, None], axis=1)
    return Z + u[:, :, None] + v[:, None, :]

def log_optimal_transport(scores: jnp.ndarray, alpha: jnp.ndarray, iters: int) -> jnp.ndarray:
    """Perform Differentiable Optimal Transport in Log-space for stability."""
    B, M, N = scores.shape
    one = jnp.ones((), dtype=scores.dtype)
    ms, ns = M * one, N * one

    bins0 = jnp.broadcast_to(alpha, (B, M, 1))
    bins1 = jnp.broadcast_to(alpha, (B, 1, N))
    alpha_corner = jnp.broadcast_to(alpha, (B, 1, 1))

    # Construct couplings matrix [B, M+1, N+1]
    row0 = jnp.concatenate([scores, bins0], axis=2)
    row1 = jnp.concatenate([bins1, alpha_corner], axis=2)
    couplings = jnp.concatenate([row0, row1], axis=1)

    norm = -jnp.log(ms + ns)
    log_mu = jnp.concatenate([jnp.broadcast_to(norm, (M,)), (jnp.log(ns) + norm)[None]])
    log_nu = jnp.concatenate([jnp.broadcast_to(norm, (N,)), (jnp.log(ms) + norm)[None]])

    log_mu = jnp.broadcast_to(log_mu, (B, M + 1))
    log_nu = jnp.broadcast_to(log_nu, (B, N + 1))

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm # multiply probabilities by M+N
    return Z

class SuperGlueJAX(nnx.Module):
    """SuperGlue feature matching middle-end in JAX/Flax NNX."""
    def __init__(self, config: Dict[str, Any] = {}, *, rngs: nnx.Rngs):
        self.config = {
            'descriptor_dim': 256,
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
            **config
        }

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'], rngs=rngs)

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'], rngs=rngs)

        self.final_proj = nnx.Linear(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            use_bias=True, rngs=rngs)

        self.bin_score = nnx.Param(jnp.array(1.0))

    def normalize_keypoints(self, kpts: jnp.ndarray, image_shape: Tuple[int, int, int, int]) -> jnp.ndarray:
        return normalize_keypoints(kpts, image_shape)

    def __call__(self, data: Dict[str, jnp.ndarray], training: bool = False) -> Dict[str, jnp.ndarray]:
        """Run SuperGlue on a pair of keypoints and descriptors."""
        desc0, desc1 = data['descriptors0'], data['descriptors1'] # [B, D, N]
        kpts0, kpts1 = data['keypoints0'], data['keypoints1'] # [B, N, 2]
        scores0, scores1 = data['scores0'], data['scores1'] # [B, N]

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0_shape'])
        kpts1 = normalize_keypoints(kpts1, data['image1_shape'])

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, scores0, training=training)
        desc1 = desc1 + self.kenc(kpts1, scores1, training=training)

        # Multi-layer Transformer network.
        desc0, desc1, attentions = self.gnn(desc0, desc1, training=training)

        # Final MLP projection.
        mdesc0 = self.final_proj(desc0.transpose(0, 2, 1)).transpose(0, 2, 1)
        mdesc1 = self.final_proj(desc1.transpose(0, 2, 1)).transpose(0, 2, 1)

        # Compute matching descriptor distance.
        scores = jnp.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / jnp.sqrt(self.config['descriptor_dim'])

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score.value,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        # scores is [B, M+1, N+1]
        matching_scores = scores[:, :-1, :-1]
        max0 = jnp.max(matching_scores, axis=2)
        indices0 = jnp.argmax(matching_scores, axis=2)
        max1 = jnp.max(matching_scores, axis=1)
        indices1 = jnp.argmax(matching_scores, axis=1)

        B, M, N = matching_scores.shape
        batch_indices = jnp.arange(B)[:, None]
        m_indices = jnp.arange(M)[None, :]
        matched_indices1 = jnp.take_along_axis(indices1, indices0, axis=1)
        mutual0 = matched_indices1 == m_indices

        mscores0 = jnp.where(mutual0, jnp.exp(max0), 0.0)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        matches0 = jnp.where(valid0, indices0, -1)

        n_indices = jnp.arange(N)[None, :]
        matched_indices0 = jnp.take_along_axis(indices0, indices1, axis=1)
        mutual1 = matched_indices0 == n_indices
        mscores1 = jnp.where(mutual1, mscores0[batch_indices, indices1], 0.0)
        valid1 = mutual1 & (mscores1 > self.config['match_threshold'])
        matches1 = jnp.where(valid1, indices1, -1)

        return {
            'matches0': matches0,
            'matches1': matches1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'attentions': attentions,
        }
