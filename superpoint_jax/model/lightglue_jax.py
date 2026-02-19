import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Dict, Optional, Any, Callable
import numpy as np

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape(x.shape[:-1] + (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape[:-2] + (-1,))

def apply_cached_rotary_emb(freqs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])

class LearnableFourierPositionalEncoding(nnx.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0, *, rngs: nnx.Rngs):
        self.gamma = gamma
        F_dim = F_dim if F_dim is not None else dim
        self.Wr = nnx.Linear(M, F_dim // 2, use_bias=False, rngs=rngs)
        # Initialization is handled during weight conversion, but we can set a default

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        projected = self.Wr(x)
        cosines, sines = jnp.cos(projected), jnp.sin(projected)
        emb = jnp.stack([cosines, sines], axis=0) # [2, B, N, F_dim//2]
        emb = jnp.expand_dims(emb, axis=2) # [2, B, 1, N, F_dim//2]
        emb = jnp.repeat(emb, 2, axis=-1) # [2, B, 1, N, F_dim]
        return emb

class TokenConfidence(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.token = nnx.Sequential(
            nnx.Linear(dim, 1, rngs=rngs),
            nnx.sigmoid
        )

    def __call__(self, desc0: jnp.ndarray, desc1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            self.token(desc0).squeeze(-1),
            self.token(desc1).squeeze(-1)
        )

class Attention(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # q, k, v: [B, H, N, D]
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            return jnp.zeros(q.shape[:-1] + (v.shape[-1],))

        scale = q.shape[-1] ** -0.5
        sim = jnp.einsum("...id,...jd->...ij", q, k) * scale

        if mask is not None:
            # mask: [B, H, N, M] or broadcastable
            sim = jnp.where(mask, sim, -1e9)

        attn = jax.nn.softmax(sim, axis=-1)
        return jnp.einsum("...ij,...jd->...id", attn, v)

class SelfBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, *, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=bias, rngs=rngs)
        self.inner_attn = Attention()
        self.out_proj = nnx.Linear(embed_dim, embed_dim, use_bias=bias, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(2 * embed_dim, 2 * embed_dim, rngs=rngs),
            nnx.LayerNorm(2 * embed_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray, encoding: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        B, N, D = x.shape
        qkv = self.Wqkv(x) # [B, N, 3*D]
        qkv = qkv.reshape(B, N, self.num_heads, -1, 3).transpose(4, 0, 2, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, N, head_dim]

        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)

        context = self.inner_attn(q, k, v, mask=mask) # [B, H, N, head_dim]
        message = self.out_proj(context.transpose(0, 2, 1, 3).reshape(B, N, D))

        return x + self.ffn(jnp.concatenate([x, message], axis=-1))

class CrossBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qk = nnx.Linear(embed_dim, embed_dim, use_bias=bias, rngs=rngs)
        self.to_v = nnx.Linear(embed_dim, embed_dim, use_bias=bias, rngs=rngs)
        self.to_out = nnx.Linear(embed_dim, embed_dim, use_bias=bias, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(2 * embed_dim, 2 * embed_dim, rngs=rngs),
            nnx.LayerNorm(2 * embed_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)
        )

    def __call__(self, x0: jnp.ndarray, x1: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, N0, D = x0.shape
        B, N1, _ = x1.shape

        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)

        qk0 = qk0.reshape(B, N0, self.num_heads, -1).transpose(0, 2, 1, 3)
        qk1 = qk1.reshape(B, N1, self.num_heads, -1).transpose(0, 2, 1, 3)
        v0 = v0.reshape(B, N0, self.num_heads, -1).transpose(0, 2, 1, 3)
        v1 = v1.reshape(B, N1, self.num_heads, -1).transpose(0, 2, 1, 3)

        qk0, qk1 = qk0 * (self.scale ** 0.5), qk1 * (self.scale ** 0.5)
        sim = jnp.einsum("bhid,bhjd->bhij", qk0, qk1)

        if mask is not None:
            sim = jnp.where(mask, sim, -1e9)

        attn01 = jax.nn.softmax(sim, axis=-1)
        attn10 = jax.nn.softmax(sim.transpose(0, 1, 3, 2), axis=-1)

        m0 = jnp.einsum("bhij,bhjd->bhid", attn01, v1)
        m1 = jnp.einsum("bhji,bhjd->bhid", attn10.transpose(0, 1, 3, 2), v0)

        m0 = self.to_out(m0.transpose(0, 2, 1, 3).reshape(B, N0, D))
        m1 = self.to_out(m1.transpose(0, 2, 1, 3).reshape(B, N1, D))

        x0 = x0 + self.ffn(jnp.concatenate([x0, m0], axis=-1))
        x1 = x1 + self.ffn(jnp.concatenate([x1, m1], axis=-1))
        return x0, x1

class TransformerLayer(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, *, rngs: nnx.Rngs):
        self.self_attn = SelfBlock(embed_dim, num_heads, bias, rngs=rngs)
        self.cross_attn = CrossBlock(embed_dim, num_heads, bias, rngs=rngs)

    def __call__(self, desc0: jnp.ndarray, desc1: jnp.ndarray,
                 encoding0: jnp.ndarray, encoding1: jnp.ndarray,
                 mask0: Optional[jnp.ndarray] = None, mask1: Optional[jnp.ndarray] = None,
                 mask01: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        desc0 = self.self_attn(desc0, encoding0, mask=mask0)
        desc1 = self.self_attn(desc1, encoding1, mask=mask1)
        desc0, desc1 = self.cross_attn(desc0, desc1, mask=mask01)
        return desc0, desc1

class MatchAssignment(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.matchability = nnx.Linear(dim, 1, use_bias=True, rngs=rngs)
        self.final_proj = nnx.Linear(dim, dim, use_bias=True, rngs=rngs)

    def __call__(self, desc0: jnp.ndarray, desc1: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        d = mdesc0.shape[-1]
        mdesc0, mdesc1 = mdesc0 / (d**0.25), mdesc1 / (d**0.25)
        sim = jnp.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)

        # log_assignment logic
        scores = self.sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def sigmoid_log_double_softmax(self, sim: jnp.ndarray, z0: jnp.ndarray, z1: jnp.ndarray) -> jnp.ndarray:
        b, m, n = sim.shape
        certainties = jax.nn.log_sigmoid(z0) + jax.nn.log_sigmoid(z1).transpose(0, 2, 1)
        scores0 = jax.nn.log_softmax(sim, axis=2)
        scores1 = jax.nn.log_softmax(sim.transpose(0, 2, 1), axis=2).transpose(0, 2, 1)

        # Build [B, M+1, N+1]
        res = jnp.zeros((b, m + 1, n + 1))
        # Top-left M x N
        res = res.at[:, :m, :n].set(scores0 + scores1 + certainties)
        # Last column
        res = res.at[:, :m, n].set(jax.nn.log_sigmoid(-z0.squeeze(-1)))
        # Last row
        res = res.at[:, m, :n].set(jax.nn.log_sigmoid(-z1.squeeze(-1)))

        return res

    def get_matchability(self, desc: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(self.matchability(desc)).squeeze(-1)

def normalize_keypoints(kpts: jnp.ndarray, size: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    if size is None:
        size = 1 + jnp.max(kpts, axis=-2) - jnp.min(kpts, axis=-2)
    # kpts: [B, N, 2], size: [B, 2]
    shift = size / 2
    scale = jnp.max(size, axis=-1, keepdims=True) / 2
    kpts = (kpts - shift[:, None, :]) / scale[:, None, :]
    return kpts

class LightGlueJAX(nnx.Module):
    def __init__(self, config: Dict[str, Any] = {}, *, rngs: nnx.Rngs):
        self.config = {
            "input_dim": 256,
            "descriptor_dim": 256,
            "add_scale_ori": False,
            "n_layers": 9,
            "num_heads": 4,
            "depth_confidence": 0.95,
            "width_confidence": 0.99,
            "filter_threshold": 0.1,
            **config
        }

        d = self.config["descriptor_dim"]
        h = self.config["num_heads"]

        if self.config["input_dim"] != d:
            self.input_proj = nnx.Linear(self.config["input_dim"], d, rngs=rngs)
        else:
            self.input_proj = lambda x: x

        head_dim = d // h
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * self.config["add_scale_ori"], head_dim, head_dim, rngs=rngs
        )

        self.transformers = nnx.List([
            TransformerLayer(d, h, rngs=rngs) for _ in range(self.config["n_layers"])
        ])

        self.log_assignment = nnx.List([
            MatchAssignment(d, rngs=rngs) for _ in range(self.config["n_layers"])
        ])

        self.token_confidence = nnx.List([
            TokenConfidence(d, rngs=rngs) for _ in range(self.config["n_layers"] - 1)
        ])

        self.confidence_thresholds = jnp.array([
            self.confidence_threshold(i) for i in range(self.config["n_layers"])
        ])

    def confidence_threshold(self, i: int) -> float:
        threshold = 0.8 + 0.1 * np.exp(-4.0 * i / self.config["n_layers"])
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(self, confidences: jnp.ndarray, scores: jnp.ndarray, layer_index: int) -> jnp.ndarray:
        """mask points which should be removed"""
        keep = scores > (1 - self.config["width_confidence"])
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(self, confidences0: jnp.ndarray, confidences1: jnp.ndarray, layer_index: int, num_points: int) -> jnp.ndarray:
        """evaluate stopping condition"""
        confidences = jnp.concatenate([confidences0, confidences1], axis=-1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - jnp.sum(confidences < threshold, axis=-1) / num_points
        return ratio_confident > self.config["depth_confidence"]

    def __call__(self, data: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        kpts0, kpts1 = data["image0"]["keypoints"], data["image1"]["keypoints"]
        desc0, desc1 = data["image0"]["descriptors"], data["image1"]["descriptors"]
        size0, size1 = data["image0"].get("image_size"), data["image1"].get("image_size")

        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape

        kpts0 = normalize_keypoints(kpts0, size0)
        kpts1 = normalize_keypoints(kpts1, size1)

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        active0 = jnp.ones((B, M), dtype=bool)
        active1 = jnp.ones((B, N), dtype=bool)

        is_stopped = jnp.zeros((B,), dtype=bool)
        final_scores = jnp.zeros((B, M + 1, N + 1))
        stop_layer = jnp.full((B,), self.config["n_layers"], dtype=jnp.int32)

        for i in range(self.config["n_layers"]):
            # Prepare masks for attention: [B, H, N, M]
            m0 = (active0[:, None, :, None] & active0[:, None, None, :])
            m1 = (active1[:, None, :, None] & active1[:, None, None, :])
            m01 = (active0[:, None, :, None] & active1[:, None, None, :])

            # Run transformer layer
            d0, d1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1,
                mask0=m0, mask1=m1, mask01=m01
            )

            # Only update if not stopped
            desc0 = jnp.where(is_stopped[:, None, None], desc0, d0)
            desc1 = jnp.where(is_stopped[:, None, None], desc1, d1)

            # Compute scores for this layer (might be the final one if stopped)
            layer_scores, _ = self.log_assignment[i](desc0, desc1)

            # If not yet stopped, this layer might be the final one
            final_scores = jnp.where(is_stopped[:, None, None], final_scores, layer_scores)

            if i < self.config["n_layers"] - 1:
                t0, t1 = None, None
                if self.config["depth_confidence"] > 0:
                    t0, t1 = self.token_confidence[i](desc0, desc1)
                    stopped_now = self.check_if_stop(t0, t1, i, M + N)

                    # Update stop_layer for elements that just stopped
                    stop_layer = jnp.where(stopped_now & ~is_stopped, i + 1, stop_layer)
                    is_stopped = is_stopped | stopped_now

                if self.config["width_confidence"] > 0:
                    # Note: Pruning should only use non-stopped descriptors
                    s0 = self.log_assignment[i].get_matchability(desc0)
                    s1 = self.log_assignment[i].get_matchability(desc1)

                    active0 = active0 & self.get_pruning_mask(t0, s0, i)
                    active1 = active1 & self.get_pruning_mask(t1, s1, i)

        # Use the collected final_scores
        scores = final_scores

        # filter_matches
        m0, m1, ms0, ms1 = self.filter_matches(scores, self.config["filter_threshold"])

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": ms0,
            "matching_scores1": ms1,
            "stop": stop_layer
        }

    def filter_matches(self, scores: jnp.ndarray, threshold: float):
        # scores: [B, M+1, N+1]
        matching_scores = scores[:, :-1, :-1]
        max0 = jnp.max(matching_scores, axis=2)
        indices0 = jnp.argmax(matching_scores, axis=2)
        max1 = jnp.max(matching_scores, axis=1)
        indices1 = jnp.argmax(matching_scores, axis=1)

        B, M, N = matching_scores.shape
        batch_indices = jnp.arange(B)[:, None]
        m_indices = jnp.arange(M)[None, :]

        # Mutual check
        matched_indices1 = jnp.take_along_axis(indices1, indices0, axis=1)
        mutual0 = matched_indices1 == m_indices

        exp_max0 = jnp.exp(max0)
        ms0 = jnp.where(mutual0, exp_max0, 0.0)
        valid0 = mutual0 & (ms0 > threshold)
        m0 = jnp.where(valid0, indices0, -1)

        # Symmetrically for m1
        n_indices = jnp.arange(N)[None, :]
        matched_indices0 = jnp.take_along_axis(indices0, indices1, axis=1)
        mutual1 = matched_indices0 == n_indices
        ms1 = jnp.where(mutual1, ms0[batch_indices, indices1], 0.0)
        valid1 = mutual1 & (ms1 > threshold)
        m1 = jnp.where(valid1, indices1, -1)

        return m0, m1, ms0, ms1
