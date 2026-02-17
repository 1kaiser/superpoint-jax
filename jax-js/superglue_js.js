/**
 * SuperGlue Feature Matching — JavaScript Inference via Safetensors Weights
 *
 * Pipeline:
 *   1. Run SuperPoint (ONNX) on two images → keypoints + descriptors
 *   2. Load SuperGlue weights from safetensors
 *   3. SuperGlue forward pass: KeypointEncoder → AttentionalGNN → Sinkhorn → Matches
 *
 * Usage:
 *   node superglue_js.js <image1> <image2> [--weights indoor|outdoor]
 *   node superglue_js.js --test   # runs on consecutive demo frames
 */

const ort = require("onnxruntime-node");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");

// ─── Safetensors Parser ────────────────────────────────────────────────────────
// Format: 8-byte LE header_size + JSON header + raw tensor data
function parseSafetensors(buffer) {
    const headerSize = Number(buffer.readBigUInt64LE(0));
    const headerJson = buffer.slice(8, 8 + Number(headerSize)).toString("utf8");
    const header = JSON.parse(headerJson);
    const dataOffset = 8 + Number(headerSize);

    const tensors = {};
    for (const [name, meta] of Object.entries(header)) {
        if (name === "__metadata__") continue;
        const { dtype, shape, data_offsets } = meta;
        const [start, end] = data_offsets;
        const rawBuf = buffer.slice(dataOffset + start, dataOffset + end);

        let typedArray;
        if (dtype === "F32") {
            typedArray = new Float32Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength / 4);
        } else if (dtype === "F64") {
            typedArray = new Float64Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength / 8);
        } else if (dtype === "I64") {
            typedArray = new BigInt64Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength / 8);
        } else if (dtype === "I32") {
            typedArray = new Int32Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength / 4);
        } else {
            console.warn(`  Unsupported dtype ${dtype} for ${name}, skipping`);
            continue;
        }

        // Make a copy to avoid alignment issues
        const data = dtype === "F32" ? new Float32Array(typedArray) :
            dtype === "F64" ? new Float64Array(typedArray) :
                dtype === "I32" ? new Int32Array(typedArray) :
                    new BigInt64Array(typedArray);

        tensors[name] = { data, shape, dtype };
    }
    return tensors;
}

// ─── SuperPoint Config ─────────────────────────────────────────────────────────
const SP_CONFIG = {
    modelPath: path.join(__dirname, "..", "superpoint.onnx"),
    stride: 8,
    nmsRadius: 4,
    detectionThreshold: 0.005,
    removeBorders: 4,
    maxNumKeypoints: 1024,
    descriptorDim: 256,
};

// ─── SuperGlue Config ──────────────────────────────────────────────────────────
const SG_CONFIG = {
    descriptorDim: 256,
    numHeads: 4,
    sinkhornIterations: 100,
    matchThreshold: 0.2,
    gnnLayers: Array.from({ length: 9 }, () => ["self", "cross"]).flat(),
    // 18 layers: [self, cross, self, cross, ...]
};

// ─── Image Loading (from SuperPoint script) ────────────────────────────────────
async function loadImage(imagePath) {
    const image = sharp(imagePath);
    const rawRgb = await image.removeAlpha().ensureAlpha(0).raw()
        .toBuffer({ resolveWithObject: true });

    const origH = rawRgb.info.height;
    const origW = rawRgb.info.width;
    const channels = rawRgb.info.channels;
    const rgbBuf = rawRgb.data;

    const grayBuf = new Uint8Array(origH * origW);
    for (let i = 0; i < origH * origW; i++) {
        const r = rgbBuf[i * channels];
        const g = rgbBuf[i * channels + 1];
        const b = rgbBuf[i * channels + 2];
        grayBuf[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    const stride = SP_CONFIG.stride;
    const padH = origH % stride === 0 ? 0 : stride - (origH % stride);
    const padW = origW % stride === 0 ? 0 : stride - (origW % stride);
    const H = origH + padH;
    const W = origW + padW;

    const float32Data = new Float32Array(H * W);
    for (let y = 0; y < origH; y++) {
        for (let x = 0; x < origW; x++) {
            float32Data[y * W + x] = grayBuf[y * origW + x] / 255.0;
        }
    }

    return { data: float32Data, H, W, origH, origW };
}

// ─── SuperPoint Post-Processing ────────────────────────────────────────────────
function pixelShuffle(scores, C, Hc, Wc, stride) {
    const H = Hc * stride, W = Wc * stride;
    const out = new Float32Array(H * W);
    for (let cy = 0; cy < Hc; cy++) {
        for (let cx = 0; cx < Wc; cx++) {
            for (let c = 0; c < C; c++) {
                const dy = Math.floor(c / stride), dx = c % stride;
                out[(cy * stride + dy) * W + (cx * stride + dx)] = scores[c * Hc * Wc + cy * Wc + cx];
            }
        }
    }
    return { data: out, H, W };
}

function maxPool2d(data, H, W, radius) {
    const out = new Float32Array(H * W);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            let maxVal = -Infinity;
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const ny = y + ky, nx = x + kx;
                    if (ny >= 0 && ny < H && nx >= 0 && nx < W)
                        maxVal = Math.max(maxVal, data[ny * W + nx]);
                }
            }
            out[y * W + x] = maxVal;
        }
    }
    return out;
}

function batchedNMS(scores, H, W, nmsRadius) {
    const pooled = maxPool2d(scores, H, W, nmsRadius);
    const maxMask = new Uint8Array(H * W);
    for (let i = 0; i < H * W; i++) maxMask[i] = scores[i] === pooled[i] ? 1 : 0;

    for (let iter = 0; iter < 2; iter++) {
        const maskFloat = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) maskFloat[i] = maxMask[i];
        const suppPooled = maxPool2d(maskFloat, H, W, nmsRadius);
        const suppMask = new Uint8Array(H * W);
        for (let i = 0; i < H * W; i++) suppMask[i] = suppPooled[i] > 0 ? 1 : 0;

        const suppScores = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) suppScores[i] = suppMask[i] ? 0 : scores[i];

        const suppPooled2 = maxPool2d(suppScores, H, W, nmsRadius);
        for (let i = 0; i < H * W; i++) {
            if (suppScores[i] === suppPooled2[i] && suppScores[i] > 0 && !suppMask[i])
                maxMask[i] = 1;
        }
    }

    const result = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) result[i] = maxMask[i] ? scores[i] : 0;
    return result;
}

function extractKeypoints(scores, H, W, threshold, maxKeypoints) {
    const candidates = [];
    for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
            const s = scores[y * W + x];
            if (s > threshold) candidates.push({ x, y, score: s });
        }
    candidates.sort((a, b) => b.score - a.score);
    if (maxKeypoints > 0) candidates.splice(maxKeypoints);
    return candidates;
}

function removeBorders(scores, H, W, pad) {
    for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
            if (y < pad || y >= H - pad || x < pad || x >= W - pad) scores[y * W + x] = -1;
    return scores;
}

function sampleDescriptors(keypoints, rawDesc, descDim, Hd, Wd, stride) {
    const descriptors = [];
    for (const kp of keypoints) {
        const sx = (kp.x / stride + 0.5) / Wd * 2 - 1;
        const sy = (kp.y / stride + 0.5) / Hd * 2 - 1;

        const px = (sx + 1) * 0.5 * (Wd - 1);
        const py = (sy + 1) * 0.5 * (Hd - 1);

        const x0 = Math.floor(px), y0 = Math.floor(py);
        const x1 = x0 + 1, y1 = y0 + 1;
        const wx = px - x0, wy = py - y0;

        const x0c = Math.max(0, Math.min(x0, Wd - 1));
        const x1c = Math.max(0, Math.min(x1, Wd - 1));
        const y0c = Math.max(0, Math.min(y0, Hd - 1));
        const y1c = Math.max(0, Math.min(y1, Hd - 1));

        const desc = new Float32Array(descDim);
        for (let c = 0; c < descDim; c++) {
            const base = c * Hd * Wd;
            const v00 = rawDesc[base + y0c * Wd + x0c];
            const v01 = rawDesc[base + y0c * Wd + x1c];
            const v10 = rawDesc[base + y1c * Wd + x0c];
            const v11 = rawDesc[base + y1c * Wd + x1c];
            desc[c] = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v01 +
                (1 - wx) * wy * v10 + wx * wy * v11;
        }

        // L2 normalize
        let norm = 0;
        for (let c = 0; c < descDim; c++) norm += desc[c] * desc[c];
        norm = Math.sqrt(norm) + 1e-8;
        for (let c = 0; c < descDim; c++) desc[c] /= norm;

        descriptors.push(desc);
    }
    return descriptors;
}

// ─── Run SuperPoint on a single image ──────────────────────────────────────────
async function runSuperPoint(session, imagePath) {
    const { data, H, W, origH, origW } = await loadImage(imagePath);
    const inputTensor = new ort.Tensor("float32", data, [1, 1, H, W]);
    const results = await session.run({ image: inputTensor });

    const rawScores = results.scores.data;
    const rawDesc = results.descriptors.data;
    const [, , Hc, Wc] = results.scores.dims;
    const [, descDim, Hd, Wd] = results.descriptors.dims;

    const { data: scoreMap, H: smH, W: smW } = pixelShuffle(rawScores, 64, Hc, Wc, SP_CONFIG.stride);
    const nmsScores = batchedNMS(scoreMap, smH, smW, SP_CONFIG.nmsRadius);
    removeBorders(nmsScores, smH, smW, SP_CONFIG.removeBorders);
    const keypoints = extractKeypoints(nmsScores, smH, smW, SP_CONFIG.detectionThreshold, SP_CONFIG.maxNumKeypoints);
    const descriptors = sampleDescriptors(keypoints, rawDesc, descDim, Hd, Wd, SP_CONFIG.stride);

    return {
        keypoints,     // [{x, y, score}, ...]
        descriptors,   // [Float32Array(256), ...]
        imageShape: [1, 1, H, W],
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SuperGlue Forward Pass (pure JS implementation)
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: Conv1d with kernel_size=1 is just a matrix multiply
// weights shape: [out_channels, in_channels], bias shape: [out_channels]
// input shape: [channels, N]  → output: [out_channels, N]
function conv1d(input, inC, N, weights, bias, outC) {
    const out = new Float32Array(outC * N);
    for (let oc = 0; oc < outC; oc++) {
        for (let n = 0; n < N; n++) {
            let sum = bias[oc];
            for (let ic = 0; ic < inC; ic++) {
                sum += weights[oc * inC + ic] * input[ic * N + n];
            }
            out[oc * N + n] = sum;
        }
    }
    return out;
}

// BatchNorm1d (eval mode): y = (x - mean) / sqrt(var + eps) * weight + bias
function batchNorm1d(input, C, N, weight, bias, runMean, runVar, eps = 1e-5) {
    const out = new Float32Array(C * N);
    for (let c = 0; c < C; c++) {
        const scale = weight[c] / Math.sqrt(runVar[c] + eps);
        const offset = bias[c] - runMean[c] * scale;
        for (let n = 0; n < N; n++) {
            out[c * N + n] = input[c * N + n] * scale + offset;
        }
    }
    return out;
}

function relu(data) {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = Math.max(0, data[i]);
    return out;
}

// MLP: sequence of Conv1d(k=1) + optional BN + ReLU (except last layer)
function mlpForward(input, inC, N, layers) {
    let x = input;
    let currentC = inC;
    for (const layer of layers) {
        if (layer.type === "conv") {
            x = conv1d(x, currentC, N, layer.weight, layer.bias, layer.outC);
            currentC = layer.outC;
        } else if (layer.type === "bn") {
            x = batchNorm1d(x, currentC, N, layer.weight, layer.bias, layer.runMean, layer.runVar);
        } else if (layer.type === "relu") {
            x = relu(x);
        }
    }
    return { data: x, channels: currentC };
}

// Normalize keypoints to [-1, 1] (matches PyTorch normalize_keypoints)
function normalizeKeypoints(kpts, imageShape) {
    const [, , height, width] = imageShape;
    const maxDim = Math.max(width, height);
    const scaling = maxDim * 0.7;
    const cx = width / 2, cy = height / 2;
    return kpts.map(kp => [(kp.x - cx) / scaling, (kp.y - cy) / scaling]);
}

// Multi-head attention: query, key, value are [d_model, N] shaped
function multiHeadAttention(query, key, value, d_model, numHeads, N_q, N_kv, projWeights, mergeWeights) {
    const dim = Math.floor(d_model / numHeads);

    // Project Q, K, V: each proj is Conv1d(d_model, d_model, k=1)
    const q = conv1d(query, d_model, N_q, projWeights[0].weight, projWeights[0].bias, d_model);
    const k = conv1d(key, d_model, N_kv, projWeights[1].weight, projWeights[1].bias, d_model);
    const v = conv1d(value, d_model, N_kv, projWeights[2].weight, projWeights[2].bias, d_model);

    // Reshape to [dim, numHeads, N] and compute attention per head
    const output = new Float32Array(d_model * N_q);

    for (let h = 0; h < numHeads; h++) {
        // For each head, extract dim-sized slice
        // scores[n_q, n_kv] = sum_d q[d,h,n_q] * k[d,h,n_kv] / sqrt(dim)
        const scale = 1.0 / Math.sqrt(dim);

        for (let nq = 0; nq < N_q; nq++) {
            // Compute attention scores for this query position
            const scores = new Float32Array(N_kv);
            for (let nkv = 0; nkv < N_kv; nkv++) {
                let dot = 0;
                for (let d = 0; d < dim; d++) {
                    const qIdx = (h * dim + d) * N_q + nq;
                    const kIdx = (h * dim + d) * N_kv + nkv;
                    dot += q[qIdx] * k[kIdx];
                }
                scores[nkv] = dot * scale;
            }

            // Softmax
            let maxScore = -Infinity;
            for (let i = 0; i < N_kv; i++) maxScore = Math.max(maxScore, scores[i]);
            let sumExp = 0;
            for (let i = 0; i < N_kv; i++) {
                scores[i] = Math.exp(scores[i] - maxScore);
                sumExp += scores[i];
            }
            for (let i = 0; i < N_kv; i++) scores[i] /= sumExp;

            // Weighted sum of values
            for (let d = 0; d < dim; d++) {
                let sum = 0;
                for (let nkv = 0; nkv < N_kv; nkv++) {
                    sum += scores[nkv] * v[(h * dim + d) * N_kv + nkv];
                }
                output[(h * dim + d) * N_q + nq] = sum;
            }
        }
    }

    // Merge: Conv1d(d_model, d_model, k=1)
    return conv1d(output, d_model, N_q, mergeWeights.weight, mergeWeights.bias, d_model);
}

// Attentional Propagation: attention + MLP residual
function attentionalPropagation(x, source, N_x, N_src, d_model, numHeads, attnWeights, mlpLayers) {
    const message = multiHeadAttention(x, source, source, d_model, numHeads, N_x, N_src,
        attnWeights.proj, attnWeights.merge);

    // Concatenate x and message: [d_model*2, N]
    const concat = new Float32Array(d_model * 2 * N_x);
    for (let c = 0; c < d_model; c++)
        for (let n = 0; n < N_x; n++) concat[c * N_x + n] = x[c * N_x + n];
    for (let c = 0; c < d_model; c++)
        for (let n = 0; n < N_x; n++) concat[(d_model + c) * N_x + n] = message[c * N_x + n];

    const { data: mlpOut } = mlpForward(concat, d_model * 2, N_x, mlpLayers);
    return mlpOut;
}

// Log-Sinkhorn iterations
function logSinkhornIterations(Z, logMu, logNu, M1, N1, iters) {
    // Z: [M1, N1], logMu: [M1], logNu: [N1]
    const u = new Float32Array(M1); // zeros
    const v = new Float32Array(N1); // zeros

    for (let iter = 0; iter < iters; iter++) {
        // u = logMu - logsumexp(Z + v[None,:], axis=1)
        for (let i = 0; i < M1; i++) {
            let maxVal = -Infinity;
            for (let j = 0; j < N1; j++) maxVal = Math.max(maxVal, Z[i * N1 + j] + v[j]);
            let sumExp = 0;
            for (let j = 0; j < N1; j++) sumExp += Math.exp(Z[i * N1 + j] + v[j] - maxVal);
            u[i] = logMu[i] - maxVal - Math.log(sumExp);
        }

        // v = logNu - logsumexp(Z + u[:,None], axis=0)
        for (let j = 0; j < N1; j++) {
            let maxVal = -Infinity;
            for (let i = 0; i < M1; i++) maxVal = Math.max(maxVal, Z[i * N1 + j] + u[i]);
            let sumExp = 0;
            for (let i = 0; i < M1; i++) sumExp += Math.exp(Z[i * N1 + j] + u[i] - maxVal);
            v[j] = logNu[j] - maxVal - Math.log(sumExp);
        }
    }

    // Z + u[:,None] + v[None,:]
    const result = new Float32Array(M1 * N1);
    for (let i = 0; i < M1; i++)
        for (let j = 0; j < N1; j++)
            result[i * N1 + j] = Z[i * N1 + j] + u[i] + v[j];
    return result;
}

// Log optimal transport with dustbins
function logOptimalTransport(scores, M, N, binScore, iters) {
    // scores: [M, N], augment to [M+1, N+1]
    const M1 = M + 1, N1 = N + 1;
    const couplings = new Float32Array(M1 * N1);

    // Fill scores
    for (let i = 0; i < M; i++)
        for (let j = 0; j < N; j++)
            couplings[i * N1 + j] = scores[i * N + j];

    // Dustbin row/col
    for (let i = 0; i < M; i++) couplings[i * N1 + N] = binScore;
    for (let j = 0; j < N; j++) couplings[M * N1 + j] = binScore;
    couplings[M * N1 + N] = binScore;

    // log_mu, log_nu
    const norm = -Math.log(M + N);
    const logMu = new Float32Array(M1);
    const logNu = new Float32Array(N1);
    for (let i = 0; i < M; i++) logMu[i] = norm;
    logMu[M] = Math.log(N) + norm;
    for (let j = 0; j < N; j++) logNu[j] = norm;
    logNu[N] = Math.log(M) + norm;

    const Z = logSinkhornIterations(couplings, logMu, logNu, M1, N1, iters);

    // Subtract norm (multiply probs by M+N)
    for (let i = 0; i < M1 * N1; i++) Z[i] -= norm;
    return { Z, M1, N1 };
}

// Extract mutual matches from assignment matrix
function extractMatches(Z, M, N, M1, N1, matchThreshold) {
    // max0[i] = argmax_j Z[i,j] for j in [0..N-1], max1[j] = argmax_i Z[i,j] for i in [0..M-1]
    const indices0 = new Int32Array(M);
    const maxScores0 = new Float32Array(M);
    for (let i = 0; i < M; i++) {
        let bestJ = 0, bestVal = Z[i * N1];
        for (let j = 1; j < N; j++) {
            if (Z[i * N1 + j] > bestVal) { bestVal = Z[i * N1 + j]; bestJ = j; }
        }
        indices0[i] = bestJ;
        maxScores0[i] = bestVal;
    }

    const indices1 = new Int32Array(N);
    const maxScores1 = new Float32Array(N);
    for (let j = 0; j < N; j++) {
        let bestI = 0, bestVal = Z[j];
        for (let i = 1; i < M; i++) {
            if (Z[i * N1 + j] > bestVal) { bestVal = Z[i * N1 + j]; bestI = i; }
        }
        indices1[j] = bestI;
        maxScores1[j] = bestVal;
    }

    // Mutual check
    const matches = [];
    for (let i = 0; i < M; i++) {
        const j = indices0[i];
        if (indices1[j] === i) {
            const score = Math.exp(maxScores0[i]);
            if (score > matchThreshold) {
                matches.push({ i, j, score });
            }
        }
    }

    return matches;
}

// ─── Load SuperGlue Weights ────────────────────────────────────────────────────
function loadSuperGlueWeights(safetensorsPath) {
    const buf = fs.readFileSync(safetensorsPath);
    const tensors = parseSafetensors(buf);

    console.log(`   Loaded ${Object.keys(tensors).length} tensors from safetensors`);

    // Build weight structures for each component
    const weights = {
        kenc: buildKencWeights(tensors),
        gnn: buildGNNWeights(tensors),
        finalProj: {
            weight: tensors["final_proj.weight"].data,
            bias: tensors["final_proj.bias"].data,
        },
        binScore: tensors["bin_score"].data[0],
    };

    return weights;
}

function buildKencWeights(tensors) {
    // kenc.encoder.0 = Conv1d(3, 32)
    // kenc.encoder.1 = BN(32)
    // kenc.encoder.2 = ReLU
    // kenc.encoder.3 = Conv1d(32, 64)
    // kenc.encoder.4 = BN(64)
    // kenc.encoder.5 = ReLU
    // kenc.encoder.6 = Conv1d(64, 128)
    // kenc.encoder.7 = BN(128)
    // kenc.encoder.8 = ReLU
    // kenc.encoder.9 = Conv1d(128, 256)
    const layers = [];
    const channelSizes = [3, 32, 64, 128, 256];

    for (let i = 0; i < 4; i++) {
        const convIdx = i * 3;
        const convKey = `kenc.encoder.${convIdx}`;
        layers.push({
            type: "conv",
            weight: tensors[`${convKey}.weight`].data,
            bias: tensors[`${convKey}.bias`].data,
            outC: channelSizes[i + 1],
        });

        if (i < 3) {
            // BN + ReLU (not on last layer)
            const bnKey = `kenc.encoder.${convIdx + 1}`;
            layers.push({
                type: "bn",
                weight: tensors[`${bnKey}.weight`].data,
                bias: tensors[`${bnKey}.bias`].data,
                runMean: tensors[`${bnKey}.running_mean`].data,
                runVar: tensors[`${bnKey}.running_var`].data,
            });
            layers.push({ type: "relu" });
        }
    }

    return layers;
}

function buildGNNWeights(tensors) {
    const numLayers = SG_CONFIG.gnnLayers.length; // 18
    const layers = [];

    for (let i = 0; i < numLayers; i++) {
        // Attention weights
        const attn = {
            proj: [0, 1, 2].map(p => ({
                weight: tensors[`gnn.layers.${i}.attn.proj.${p}.weight`].data,
                bias: tensors[`gnn.layers.${i}.attn.proj.${p}.bias`].data,
            })),
            merge: {
                weight: tensors[`gnn.layers.${i}.attn.merge.weight`].data,
                bias: tensors[`gnn.layers.${i}.attn.merge.bias`].data,
            },
        };

        // MLP weights: mlp.0 = Conv1d(512, 512), mlp.1 = BN(512), mlp.2 = ReLU, mlp.3 = Conv1d(512, 256)
        const mlpLayers = [];
        // Conv1d(512, 512)
        mlpLayers.push({
            type: "conv",
            weight: tensors[`gnn.layers.${i}.mlp.0.weight`].data,
            bias: tensors[`gnn.layers.${i}.mlp.0.bias`].data,
            outC: 512,
        });
        mlpLayers.push({
            type: "bn",
            weight: tensors[`gnn.layers.${i}.mlp.1.weight`].data,
            bias: tensors[`gnn.layers.${i}.mlp.1.bias`].data,
            runMean: tensors[`gnn.layers.${i}.mlp.1.running_mean`].data,
            runVar: tensors[`gnn.layers.${i}.mlp.1.running_var`].data,
        });
        mlpLayers.push({ type: "relu" });
        // Conv1d(512, 256)
        mlpLayers.push({
            type: "conv",
            weight: tensors[`gnn.layers.${i}.mlp.3.weight`].data,
            bias: tensors[`gnn.layers.${i}.mlp.3.bias`].data,
            outC: 256,
        });

        layers.push({ attn, mlp: mlpLayers, name: SG_CONFIG.gnnLayers[i] });
    }

    return layers;
}

// ─── SuperGlue Forward Pass ────────────────────────────────────────────────────
function superGlueForward(kp0, desc0, imgShape0, kp1, desc1, imgShape1, weights) {
    const d = SG_CONFIG.descriptorDim;
    const M = kp0.length;
    const N = kp1.length;

    if (M === 0 || N === 0) {
        return { matches: [], M, N };
    }

    // 1. Normalize keypoints
    const normKp0 = normalizeKeypoints(kp0, imgShape0);
    const normKp1 = normalizeKeypoints(kp1, imgShape1);

    // 2. Prepare descriptor matrices: [256, M] and [256, N]
    const d0 = new Float32Array(d * M);
    const d1 = new Float32Array(d * N);
    for (let i = 0; i < M; i++)
        for (let c = 0; c < d; c++) d0[c * M + i] = desc0[i][c];
    for (let i = 0; i < N; i++)
        for (let c = 0; c < d; c++) d1[c * N + i] = desc1[i][c];

    // 3. Keypoint encoder input: [3, M] = [x_norm, y_norm, score]
    const kpInput0 = new Float32Array(3 * M);
    const kpInput1 = new Float32Array(3 * N);
    for (let i = 0; i < M; i++) {
        kpInput0[0 * M + i] = normKp0[i][0]; // x
        kpInput0[1 * M + i] = normKp0[i][1]; // y
        kpInput0[2 * M + i] = kp0[i].score;
    }
    for (let i = 0; i < N; i++) {
        kpInput1[0 * N + i] = normKp1[i][0];
        kpInput1[1 * N + i] = normKp1[i][1];
        kpInput1[2 * N + i] = kp1[i].score;
    }

    // 4. Keypoint encoding: desc += kenc(kpts, scores)
    const { data: kenc0 } = mlpForward(kpInput0, 3, M, weights.kenc);
    const { data: kenc1 } = mlpForward(kpInput1, 3, N, weights.kenc);

    // Add to descriptors
    let x0 = new Float32Array(d * M);
    let x1 = new Float32Array(d * N);
    for (let i = 0; i < d * M; i++) x0[i] = d0[i] + kenc0[i];
    for (let i = 0; i < d * N; i++) x1[i] = d1[i] + kenc1[i];

    // 5. Attentional GNN: 18 layers
    for (let l = 0; l < weights.gnn.length; l++) {
        const layer = weights.gnn[l];
        let src0, src1;
        if (layer.name === "cross") {
            src0 = x1; src1 = x0;
        } else {
            src0 = x0; src1 = x1;
        }

        const delta0 = attentionalPropagation(x0, src0, M, layer.name === "cross" ? N : M,
            d, SG_CONFIG.numHeads, layer.attn, layer.mlp);
        const delta1 = attentionalPropagation(x1, src1, N, layer.name === "cross" ? M : N,
            d, SG_CONFIG.numHeads, layer.attn, layer.mlp);

        const newX0 = new Float32Array(d * M);
        const newX1 = new Float32Array(d * N);
        for (let i = 0; i < d * M; i++) newX0[i] = x0[i] + delta0[i];
        for (let i = 0; i < d * N; i++) newX1[i] = x1[i] + delta1[i];
        x0 = newX0;
        x1 = newX1;

        if ((l + 1) % 6 === 0) {
            process.stdout.write(`   GNN layer ${l + 1}/${weights.gnn.length}\r`);
        }
    }
    console.log(`   GNN: ${weights.gnn.length} layers complete        `);

    // 6. Final projection
    const mdesc0 = conv1d(x0, d, M, weights.finalProj.weight, weights.finalProj.bias, d);
    const mdesc1 = conv1d(x1, d, N, weights.finalProj.weight, weights.finalProj.bias, d);

    // 7. Score matrix: scores[i,j] = sum_c mdesc0[c,i] * mdesc1[c,j] / sqrt(d)
    const scores = new Float32Array(M * N);
    const scale = 1.0 / Math.sqrt(d);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let dot = 0;
            for (let c = 0; c < d; c++) dot += mdesc0[c * M + i] * mdesc1[c * N + j];
            scores[i * N + j] = dot * scale;
        }
    }

    // 8. Log optimal transport
    const { Z, M1, N1 } = logOptimalTransport(scores, M, N, weights.binScore, SG_CONFIG.sinkhornIterations);

    // 9. Extract matches
    const matches = extractMatches(Z, M, N, M1, N1, SG_CONFIG.matchThreshold);

    return { matches, M, N };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════
async function main() {
    const args = process.argv.slice(2);

    let img1Path, img2Path, weightsName = "indoor";

    if (args[0] === "--test") {
        // Use consecutive demo frames
        const framesDir = path.join(__dirname, "..", "demo", "frames", "input_frames");
        if (!fs.existsSync(framesDir)) {
            console.error(`Frame directory not found: ${framesDir}`);
            process.exit(1);
        }
        img1Path = path.join(framesDir, "frame_0000.png");
        img2Path = path.join(framesDir, "frame_0001.png");
        console.log("Using demo frames: frame_0000.png & frame_0001.png");
    } else if (args.length >= 2) {
        img1Path = args[0];
        img2Path = args[1];
        if (args.includes("--outdoor")) weightsName = "outdoor";
    } else {
        console.log("Usage: node superglue_js.js <image1> <image2> [--weights indoor|outdoor]");
        console.log("       node superglue_js.js --test");
        process.exit(0);
    }

    console.log("═".repeat(60));
    console.log("  SuperGlue.js — Feature Matching via Safetensors Weights");
    console.log("═".repeat(60));

    // 1. Load SuperPoint model
    console.log("\n📦 Loading SuperPoint ONNX model...");
    const t0 = performance.now();
    const spSession = await ort.InferenceSession.create(SP_CONFIG.modelPath);
    console.log(`   SuperPoint loaded in ${(performance.now() - t0).toFixed(0)}ms`);

    // 2. Run SuperPoint on both images
    console.log("\n⚡ Running SuperPoint on image 1...");
    const t1 = performance.now();
    const sp1 = await runSuperPoint(spSession, img1Path);
    console.log(`   Image 1: ${sp1.keypoints.length} keypoints (${(performance.now() - t1).toFixed(0)}ms)`);

    console.log("\n⚡ Running SuperPoint on image 2...");
    const t2 = performance.now();
    const sp2 = await runSuperPoint(spSession, img2Path);
    console.log(`   Image 2: ${sp2.keypoints.length} keypoints (${(performance.now() - t2).toFixed(0)}ms)`);

    // 3. Load SuperGlue weights
    const sgPath = path.join(__dirname, "..", "models", "weights", `superglue_${weightsName}.safetensors`);

    console.log(`\n🔗 Loading SuperGlue (${weightsName}) weights...`);
    const t3 = performance.now();
    const sgWeights = loadSuperGlueWeights(sgPath);
    console.log(`   Weights loaded in ${(performance.now() - t3).toFixed(0)}ms`);
    console.log(`   bin_score: ${sgWeights.binScore.toFixed(4)}`);

    // 4. Run SuperGlue matching
    console.log("\n🔗 Running SuperGlue matching...");
    const t4 = performance.now();
    const result = superGlueForward(
        sp1.keypoints, sp1.descriptors, sp1.imageShape,
        sp2.keypoints, sp2.descriptors, sp2.imageShape,
        sgWeights
    );
    const matchMs = performance.now() - t4;
    console.log(`   Matching: ${matchMs.toFixed(0)}ms`);

    // 5. Results
    console.log("\n" + "─".repeat(60));
    console.log("📊 Results Summary");
    console.log("─".repeat(60));
    console.log(`   Image 1 keypoints: ${sp1.keypoints.length}`);
    console.log(`   Image 2 keypoints: ${sp2.keypoints.length}`);
    console.log(`   Matches found:     ${result.matches.length}`);
    if (result.matches.length > 0) {
        const scores = result.matches.map(m => m.score);
        console.log(`   Match score range: [${Math.min(...scores).toFixed(4)}, ${Math.max(...scores).toFixed(4)}]`);
        console.log(`   Mean match score:  ${(scores.reduce((a, b) => a + b) / scores.length).toFixed(4)}`);

        console.log(`\n   Top 10 matches:`);
        console.log(`   ${"#".padStart(3)}  ${"kp1_x".padStart(6)}  ${"kp1_y".padStart(6)}  ${"kp2_x".padStart(6)}  ${"kp2_y".padStart(6)}  ${"score".padStart(8)}`);
        console.log(`   ${"─".repeat(45)}`);
        const top = result.matches.sort((a, b) => b.score - a.score).slice(0, 10);
        for (let i = 0; i < top.length; i++) {
            const m = top[i];
            const k1 = sp1.keypoints[m.i], k2 = sp2.keypoints[m.j];
            console.log(`   ${(i + 1 + "").padStart(3)}  ${(k1.x + "").padStart(6)}  ${(k1.y + "").padStart(6)}  ${(k2.x + "").padStart(6)}  ${(k2.y + "").padStart(6)}  ${m.score.toFixed(6).padStart(8)}`);
        }
    }

    const totalMs = performance.now() - t0;
    console.log(`\n   Total time: ${totalMs.toFixed(0)}ms`);
    console.log("═".repeat(60));
}

main().catch(console.error);
