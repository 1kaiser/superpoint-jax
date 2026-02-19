# Section 4: Run Comparison on Real Data
dataset_path = ROOT_DIR / 'demo/frames/input_frames/'
frames = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])

if len(frames) > 10:
    idx0 = random.randint(0, len(frames) - 11)
    idx1 = idx0 + 10
    img_ref_path = dataset_path / frames[idx0]
    img_target_path = dataset_path / frames[idx1]
    title_suffix = f" (Gap 10: {frames[idx0]} vs {frames[idx1]})"
else:
    # Fallback to synthetic if something went wrong
    img_ref = np.zeros((480, 640), dtype=np.uint8)
    for _ in range(50):
        x, y = random.randint(50, 590), random.randint(50, 430)
        cv2.circle(img_ref, (x, y), 3, 255, -1)
    img_target = cv2.warpAffine(img_ref, cv2.getRotationMatrix2D((320, 240), 2, 1.0), (640, 480))
    img_ref_float = img_ref.astype(np.float32) / 255.0
    img_target_float = img_target.astype(np.float32) / 255.0
    title_suffix = " (Synthetic)"

img_ref, img_ref_float = load_image(img_ref_path)
img_target, img_target_float = load_image(img_target_path)

print(f"Running Comparison on real frames: {frames[idx0]} and {frames[idx1]}")

print("Running LightGlue JAX...")
res_jax_lg = run_jax_lg(img_ref_float, img_target_float)
visualize_matches(img_ref, img_target, res_jax_lg, "LightGlue JAX" + title_suffix)

print("Running SuperGlue JAX...")
res_jax_sg = run_jax_sg(img_ref_float, img_target_float)
visualize_matches(img_ref, img_target, res_jax_sg, "SuperGlue JAX" + title_suffix)

print("Running LightGlue PyTorch...")
res_torch_lg = run_pytorch_lg(img_ref_float, img_target_float)
visualize_matches(img_ref, img_target, res_torch_lg, "LightGlue PyTorch" + title_suffix)
