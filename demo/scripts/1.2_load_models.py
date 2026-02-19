# SuperPoint Models
sp_torch = SuperPointTorch(max_num_keypoints=1024).to(device)
weights_path = ROOT_DIR / 'weights/superpoint_torch.pth'
sp_torch.load_state_dict(torch.load(str(weights_path), map_location=device))
sp_torch.eval()

sp_jax = SuperPointJAX(max_num_keypoints=1024, rngs=nnx.Rngs(0))
sp_jax = convert_superpoint_weights(sp_torch, sp_jax)

# SuperGlue Models
sg_torch = SuperGlueTorch({'weights': 'indoor'}).to(device)
sg_torch.eval()

sg_jax = SuperGlueJAX(rngs=nnx.Rngs(0))
sg_jax = convert_superglue_weights(sg_torch, sg_jax)

# LightGlue Model (PyTorch)
lg_torch = LightGlue(features='superpoint').to(device)
lg_torch.eval()

# LightGlue Model (JAX)
lg_jax = LightGlueJAX(rngs=nnx.Rngs(0))
lg_jax = convert_lightglue_weights(lg_torch, lg_jax)

print("All models loaded and converted.")
