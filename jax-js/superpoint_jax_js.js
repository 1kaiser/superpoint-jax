/**
 * SuperPoint Keypoint Detection — JavaScript Inference via ONNX Runtime
 *
 * Mirrors the Python SuperPointTorch.forward() pipeline:
 *   1. Load image → grayscale → normalize to [0,1]
 *   2. Run ONNX model (backbone + detector + descriptor heads)
 *   3. Post-process: pixel-shuffle scores, NMS, border removal, threshold
 *   4. Extract keypoints, scores, and bilinear-sampled descriptors
 *
 * Usage:
 *   node superpoint_jax_js.js <image_path>
 *   node superpoint_jax_js.js --test   # runs on a synthetic test pattern
 */

const ort = require("onnxruntime-node");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");
const zlib = require("zlib");

// ─── NPZ File Reader (minimal, for float32/float64 arrays) ─────────────────
async function loadNpz(npzPath) {
    const AdmZip = await loadAdmZip();
    const zip = new AdmZip(npzPath);
    const entries = zip.getEntries();
    const arrays = {};

    for (const entry of entries) {
        const name = entry.entryName.replace(".npy", "");
        const buf = entry.getData();
        arrays[name] = parseNpy(buf);
    }
    return arrays;
}

function parseNpy(buffer) {
    // NumPy .npy format: magic + version + header_len + header + data
    const magic = buffer.slice(0, 6).toString();
    const headerLen = buffer.readUInt16LE(8);
    const header = buffer.slice(10, 10 + headerLen).toString();

    // Parse dtype and shape from header
    const dtypeMatch = header.match(/'descr':\s*'([^']+)'/);
    const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
    const dtype = dtypeMatch ? dtypeMatch[1] : "<f4";
    const shape = shapeMatch
        ? shapeMatch[1]
            .split(",")
            .filter((s) => s.trim())
            .map(Number)
        : [];

    const dataOffset = 10 + headerLen;
    const data = buffer.slice(dataOffset);

    // Convert to Float32Array or Float64Array
    if (dtype.includes("f8")) {
        return new Float64Array(
            data.buffer,
            data.byteOffset,
            data.byteLength / 8
        );
    }
    return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
}

async function loadAdmZip() {
    // Lazy-load adm-zip (install if needed)
    try {
        return require("adm-zip");
    } catch {
        console.log("   Installing adm-zip...");
        const { execSync } = require("child_process");
        execSync("npm install adm-zip --save", { cwd: __dirname, stdio: "pipe" });
        return require("adm-zip");
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────
const CONFIG = {
    modelPath: path.join(__dirname, "..", "superpoint.onnx"),
    nmsRadius: 4,
    detectionThreshold: 0.005,
    maxNumKeypoints: null, // null = unlimited, matching PyTorch default
    removeBorders: 4,
    stride: 8, // 2^(num_pool_layers) = 2^3 = 8
    descriptorDim: 256,
};

// ─── Utility: 2D Max Pool (stride=1, same padding) ─────────────────────────
function maxPool2d(data, H, W, radius) {
    const k = 2 * radius + 1;
    const out = new Float32Array(H * W);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            let maxVal = -Infinity;
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const ny = y + ky;
                    const nx = x + kx;
                    if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                        maxVal = Math.max(maxVal, data[ny * W + nx]);
                    }
                }
            }
            out[y * W + x] = maxVal;
        }
    }
    return out;
}

// ─── Pixel Shuffle: (64, H/8, W/8) → (H, W) ──────────────────────────────
function pixelShuffle(scores, C, Hc, Wc, stride) {
    // scores layout: C channels of Hc×Wc
    // C = stride^2 = 64, each channel maps to a position in the stride×stride block
    const H = Hc * stride;
    const W = Wc * stride;
    const out = new Float32Array(H * W);

    for (let cy = 0; cy < Hc; cy++) {
        for (let cx = 0; cx < Wc; cx++) {
            for (let c = 0; c < C; c++) {
                // PyTorch layout: reshape(b, h, w, stride, stride) then permute(b, h, stride, w, stride)
                // Channel c maps to dy = floor(c / stride), dx = c % stride
                const dy = Math.floor(c / stride);
                const dx = c % stride;
                const oy = cy * stride + dy;
                const ox = cx * stride + dx;
                out[oy * W + ox] = scores[c * Hc * Wc + cy * Wc + cx];
            }
        }
    }
    return { data: out, H, W };
}

// ─── Non-Maximum Suppression ────────────────────────────────────────────────
function batchedNMS(scores, H, W, nmsRadius) {
    const zeros = new Float32Array(H * W);

    // First pass: local maxima
    const pooled = maxPool2d(scores, H, W, nmsRadius);
    const maxMask = new Uint8Array(H * W);
    for (let i = 0; i < H * W; i++) {
        maxMask[i] = scores[i] === pooled[i] ? 1 : 0;
    }

    // Two iterations of suppression (matching PyTorch implementation)
    for (let iter = 0; iter < 2; iter++) {
        // Pool the mask to get suppression region
        const maskFloat = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) maskFloat[i] = maxMask[i];
        const suppPooled = maxPool2d(maskFloat, H, W, nmsRadius);
        const suppMask = new Uint8Array(H * W);
        for (let i = 0; i < H * W; i++) suppMask[i] = suppPooled[i] > 0 ? 1 : 0;

        // Suppress scores where suppMask is active
        const suppScores = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) {
            suppScores[i] = suppMask[i] ? 0 : scores[i];
        }

        // Find new maxima in suppressed scores
        const suppPooled2 = maxPool2d(suppScores, H, W, nmsRadius);
        for (let i = 0; i < H * W; i++) {
            const isNewMax = suppScores[i] === suppPooled2[i] && suppScores[i] > 0;
            if (isNewMax && !suppMask[i]) {
                maxMask[i] = 1;
            }
        }
    }

    // Apply mask
    const result = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
        result[i] = maxMask[i] ? scores[i] : 0;
    }
    return result;
}

// ─── Remove Border Keypoints ────────────────────────────────────────────────
function removeBorders(scores, H, W, pad) {
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (y < pad || y >= H - pad || x < pad || x >= W - pad) {
                scores[y * W + x] = -1;
            }
        }
    }
    return scores;
}

// ─── Extract Keypoints Above Threshold ──────────────────────────────────────
function extractKeypoints(scores, H, W, threshold, maxKeypoints) {
    const candidates = [];
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const s = scores[y * W + x];
            if (s > threshold) {
                candidates.push({ x, y, score: s });
            }
        }
    }

    // Sort by score descending
    candidates.sort((a, b) => b.score - a.score);

    // Top-k
    const topK =
        maxKeypoints != null
            ? candidates.slice(0, maxKeypoints)
            : candidates;

    return topK;
}

// ─── Bilinear Descriptor Sampling ───────────────────────────────────────────
function sampleDescriptors(keypoints, descriptors, C, Hd, Wd, stride) {
    // descriptors: (C, Hd, Wd) in NCHW format
    // keypoints: [{x, y}] in pixel coordinates
    // Normalize keypoint coords to descriptor map space
    const result = [];

    for (const kp of keypoints) {
        // Map pixel coords to descriptor grid coords
        const gx = (kp.x + 0.5) / (Wd * stride) * 2 - 1; // normalize to [-1, 1]
        const gy = (kp.y + 0.5) / (Hd * stride) * 2 - 1;

        // Map from [-1,1] to [0, Wd-1] and [0, Hd-1] for bilinear interp
        const fx = ((gx + 1) / 2) * (Wd - 1);
        const fy = ((gy + 1) / 2) * (Hd - 1);

        const x0 = Math.floor(fx);
        const y0 = Math.floor(fy);
        const x1 = Math.min(x0 + 1, Wd - 1);
        const y1 = Math.min(y0 + 1, Hd - 1);

        const wx = fx - x0;
        const wy = fy - y0;

        // Bilinear interpolation for each channel
        const desc = new Float32Array(C);
        let norm = 0;
        for (let c = 0; c < C; c++) {
            const offset = c * Hd * Wd;
            const v00 = descriptors[offset + y0 * Wd + x0];
            const v01 = descriptors[offset + y0 * Wd + x1];
            const v10 = descriptors[offset + y1 * Wd + x0];
            const v11 = descriptors[offset + y1 * Wd + x1];
            const val =
                v00 * (1 - wx) * (1 - wy) +
                v01 * wx * (1 - wy) +
                v10 * (1 - wx) * wy +
                v11 * wx * wy;
            desc[c] = val;
            norm += val * val;
        }

        // L2 normalize
        norm = Math.sqrt(norm + 1e-8);
        for (let c = 0; c < C; c++) desc[c] /= norm;

        result.push(desc);
    }

    return result;
}

// ─── Load and Preprocess Image ──────────────────────────────────────────────
async function loadImage(imagePath) {
    const image = sharp(imagePath);

    // Use native grayscale for performance
    const { data: grayBuf, info } = await image
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const origH = info.height;
    const origW = info.width;

    // Zero-pad to multiple of stride (matching np.pad mode='constant')
    const stride = CONFIG.stride;
    const padH = origH % stride === 0 ? 0 : stride - (origH % stride);
    const padW = origW % stride === 0 ? 0 : stride - (origW % stride);
    const H = origH + padH;
    const W = origW + padW;

    // Create padded float32 array (default 0.0 = zero-padding)
    const float32Data = new Float32Array(H * W); // initialized to 0
    for (let y = 0; y < origH; y++) {
        for (let x = 0; x < origW; x++) {
            float32Data[y * W + x] = grayBuf[y * origW + x] / 255.0;
        }
    }

    return {
        data: float32Data,
        H,
        W,
        origH,
        origW,
    };
}

// ─── Generate Test Pattern ──────────────────────────────────────────────────
function generateTestPattern(H = 240, W = 320) {
    const data = new Float32Array(H * W);
    // Checkerboard + gaussian blobs for keypoint features
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const checker = ((Math.floor(x / 32) + Math.floor(y / 32)) % 2) * 0.3;
            // Add some corner-like features
            const cx1 = 80, cy1 = 60;
            const cx2 = 240, cy2 = 180;
            const d1 = Math.exp(
                -((x - cx1) ** 2 + (y - cy1) ** 2) / (2 * 20 * 20)
            );
            const d2 = Math.exp(
                -((x - cx2) ** 2 + (y - cy2) ** 2) / (2 * 20 * 20)
            );
            data[y * W + x] = Math.min(1.0, checker + d1 * 0.7 + d2 * 0.7);
        }
    }
    return { data, H, W, origH: H, origW: W };
}

// ─── Main Inference Pipeline ────────────────────────────────────────────────
async function runInference(imagePath) {
    console.log("═".repeat(60));
    console.log("  SuperPoint.js — Keypoint Detection via ONNX Runtime");
    console.log("═".repeat(60));

    // 1. Load model
    console.log("\n📦 Loading ONNX model...");
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(CONFIG.modelPath);
    console.log(`   Model loaded in ${(performance.now() - t0).toFixed(0)}ms`);
    console.log(`   Inputs:  ${session.inputNames}`);
    console.log(`   Outputs: ${session.outputNames}`);

    // 2. Load / generate image
    let imageInfo;
    if (imagePath === "--test") {
        console.log("\n🎨 Generating test pattern (240×320)...");
        imageInfo = generateTestPattern();
    } else {
        console.log(`\n🖼️  Loading image: ${imagePath}`);
        imageInfo = await loadImage(imagePath);
    }
    const { data: imageData, H: imgH, W: imgW } = imageInfo;
    console.log(`   Size: ${imgW}×${imgH} (padded to stride ${CONFIG.stride})`);

    // 3. Run ONNX inference
    console.log("\n⚡ Running model inference...");
    const inputTensor = new ort.Tensor("float32", imageData, [1, 1, imgH, imgW]);
    const t1 = performance.now();
    const results = await session.run({ image: inputTensor });
    const inferenceMs = performance.now() - t1;
    console.log(`   Inference time: ${inferenceMs.toFixed(1)}ms`);

    const rawScores = results.scores.data; // (1, 64, H/8, W/8)
    const rawDesc = results.descriptors.data; // (1, 256, H/8, W/8)
    const scoreDims = results.scores.dims; // [1, 64, Hc, Wc]
    const descDims = results.descriptors.dims; // [1, 256, Hd, Wd]

    const Hc = scoreDims[2];
    const Wc = scoreDims[3];
    const Hd = descDims[2];
    const Wd = descDims[3];
    console.log(`   Raw scores:      [${scoreDims}]`);
    console.log(`   Raw descriptors: [${descDims}]`);

    // 4. Post-processing
    console.log("\n🔧 Post-processing...");
    const t2 = performance.now();

    // 4a. Pixel shuffle: (64, Hc, Wc) → (H, W) score map
    const { data: scoreMap, H: smH, W: smW } = pixelShuffle(
        rawScores,
        64,
        Hc,
        Wc,
        CONFIG.stride
    );
    console.log(`   Pixel shuffle: (64,${Hc},${Wc}) → (${smH},${smW})`);

    // 4b. Non-maximum suppression
    const nmsScores = batchedNMS(scoreMap, smH, smW, CONFIG.nmsRadius);
    console.log(`   NMS (radius=${CONFIG.nmsRadius}) applied`);

    // 4c. Remove border keypoints
    if (CONFIG.removeBorders > 0) {
        removeBorders(nmsScores, smH, smW, CONFIG.removeBorders);
        console.log(`   Border removal (pad=${CONFIG.removeBorders}) applied`);
    }

    // 4d. Extract keypoints
    const keypoints = extractKeypoints(
        nmsScores,
        smH,
        smW,
        CONFIG.detectionThreshold,
        CONFIG.maxNumKeypoints
    );
    console.log(`   Keypoints detected: ${keypoints.length}`);

    // 4e. Sample descriptors at keypoint locations
    const descriptors = sampleDescriptors(
        keypoints,
        rawDesc,
        CONFIG.descriptorDim,
        Hd,
        Wd,
        CONFIG.stride
    );
    const postMs = performance.now() - t2;
    console.log(`   Post-processing time: ${postMs.toFixed(1)}ms`);

    // 5. Results summary
    console.log("\n" + "─".repeat(60));
    console.log("📊 Results Summary");
    console.log("─".repeat(60));
    console.log(`   Total keypoints:    ${keypoints.length}`);
    if (keypoints.length > 0) {
        const scores = keypoints.map((k) => k.score);
        console.log(
            `   Score range:        [${Math.min(...scores).toFixed(4)}, ${Math.max(...scores).toFixed(4)}]`
        );
        console.log(
            `   Mean score:         ${(scores.reduce((a, b) => a + b) / scores.length).toFixed(4)}`
        );
        console.log(`   Descriptor dim:     ${CONFIG.descriptorDim}`);

        // Print top 10 keypoints
        console.log(`\n   Top ${Math.min(10, keypoints.length)} keypoints:`);
        console.log("   " + "─".repeat(40));
        console.log("    #    x      y      score");
        console.log("   " + "─".repeat(40));
        const top10 = keypoints.slice(0, 10);
        top10.forEach((kp, i) => {
            console.log(
                `   ${String(i + 1).padStart(3)}  ${String(kp.x).padStart(5)}  ${String(kp.y).padStart(5)}  ${kp.score.toFixed(6)}`
            );
        });

        // Verify descriptor norms (should be ~1.0)
        if (descriptors.length > 0) {
            const norms = descriptors.map((d) =>
                Math.sqrt(d.reduce((s, v) => s + v * v, 0))
            );
            console.log(
                `\n   Descriptor L2 norm: [${Math.min(...norms).toFixed(4)}, ${Math.max(...norms).toFixed(4)}] (should be ~1.0)`
            );
        }
    }

    console.log(`\n   Total time:         ${(inferenceMs + postMs).toFixed(1)}ms`);
    console.log("═".repeat(60));

    // ── Compare with Python ground truth (if available) ──────────────
    const gtPath = path.join(__dirname, "..", "demo", "pytorch_ground_truth.npz");
    if (fs.existsSync(gtPath)) {
        console.log("\n" + "═".repeat(60));
        console.log("🔬 Comparison with PyTorch Ground Truth");
        console.log("═".repeat(60));
        try {
            const npzData = await loadNpz(gtPath);
            const pyKpts = npzData.keypoints; // [[x,y], ...]
            const pyScores = npzData.scores;  // [score, ...]
            const pyCount = pyScores.length;

            console.log(`\n   PyTorch keypoints:  ${pyCount}`);
            console.log(`   JS keypoints:      ${keypoints.length}`);
            console.log(`   Difference:        ${Math.abs(pyCount - keypoints.length)} (${((Math.abs(pyCount - keypoints.length) / pyCount) * 100).toFixed(1)}%)`);

            // Compare top scores
            const pyTopScores = Array.from(pyScores).sort((a, b) => b - a).slice(0, 5);
            const jsTopScores = keypoints.slice(0, 5).map(k => k.score);
            console.log(`\n   Top-5 scores comparison:`);
            console.log(`   ${'#'.padStart(3)}  ${'PyTorch'.padStart(10)}  ${'JS'.padStart(10)}  ${'Δ'.padStart(10)}`);
            console.log(`   ${'─'.repeat(40)}`);
            for (let i = 0; i < 5; i++) {
                const py = pyTopScores[i] || 0;
                const js = jsTopScores[i] || 0;
                const diff = Math.abs(py - js);
                console.log(`   ${(i + 1 + '').padStart(3)}  ${py.toFixed(6).padStart(10)}  ${js.toFixed(6).padStart(10)}  ${diff.toFixed(6).padStart(10)}`);
            }

            // Find common keypoints (within 2px tolerance)
            const tolerance = 2;
            let commonCount = 0;
            for (const jkp of keypoints) {
                for (let pi = 0; pi < pyKpts.length; pi += 2) {
                    const px = pyKpts[pi], py2 = pyKpts[pi + 1];
                    if (Math.abs(jkp.x - px) <= tolerance && Math.abs(jkp.y - py2) <= tolerance) {
                        commonCount++;
                        break;
                    }
                }
            }
            const matchRate = (commonCount / Math.min(keypoints.length, pyCount) * 100).toFixed(1);
            console.log(`\n   Common keypoints (±${tolerance}px):  ${commonCount}/${Math.min(keypoints.length, pyCount)} (${matchRate}%)`);

        } catch (e) {
            console.log(`   ⚠️  Could not compare: ${e.message}`);
        }
        console.log("═".repeat(60));
    }

    return { keypoints, descriptors };
}

// ─── Entry Point ────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const imagePath = args[0] || "--test";

if (!fs.existsSync(CONFIG.modelPath)) {
    console.error(`\n❌ ONNX model not found at: ${CONFIG.modelPath}`);
    console.error("\n   Run the export script first:");
    console.error("   conda run -n num_python python export_to_onnx.py\n");
    process.exit(1);
}

if (imagePath !== "--test" && !fs.existsSync(imagePath)) {
    console.error(`\n❌ Image not found: ${imagePath}`);
    process.exit(1);
}

runInference(imagePath).catch((err) => {
    console.error("\n❌ Inference failed:", err);
    process.exit(1);
});
