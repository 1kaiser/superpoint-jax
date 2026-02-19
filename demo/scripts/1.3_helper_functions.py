def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    img_float = img.astype(np.float32) / 255.0
    return img, img_float

def run_jax_lg(img0_float, img1_float):
    inp0 = jnp.array(img0_float)[None, ..., None]
    inp1 = jnp.array(img1_float)[None, ..., None]

    out0 = sp_jax(inp0, training=False)
    out1 = sp_jax(inp1, training=False)

    v0 = int(out0['valid_counts'][0])
    v1 = int(out1['valid_counts'][0])

    data = {
        'image0': {
            'keypoints': out0['keypoints'][:, :v0],
            'descriptors': out0['descriptors'][:, :v0],
            'image_size': jnp.array([[img0_float.shape[1], img0_float.shape[0]]])
        },
        'image1': {
            'keypoints': out1['keypoints'][:, :v1],
            'descriptors': out1['descriptors'][:, :v1],
            'image_size': jnp.array([[img1_float.shape[1], img1_float.shape[0]]])
        }
    }

    res = lg_jax(data)
    return {
        'kpts0': np.array(data['image0']['keypoints'][0]),
        'kpts1': np.array(data['image1']['keypoints'][0]),
        'matches0': np.array(res['matches0'][0]),
        'matching_scores0': np.array(res['matching_scores0'][0]),
    }

def run_jax_sg(img0_float, img1_float):
    inp0 = jnp.array(img0_float)[None, ..., None]
    inp1 = jnp.array(img1_float)[None, ..., None]

    out0 = sp_jax(inp0, training=False)
    out1 = sp_jax(inp1, training=False)

    v0 = int(out0['valid_counts'][0])
    v1 = int(out1['valid_counts'][0])

    data = {
        'keypoints0': out0['keypoints'][:, :v0],
        'scores0': out0['scores'][:, :v0],
        'descriptors0': out0['descriptors'][:, :v0].transpose(0, 2, 1),
        'image0_shape': (1, 1, *img0_float.shape),
        'keypoints1': out1['keypoints'][:, :v1],
        'scores1': out1['scores'][:, :v1],
        'descriptors1': out1['descriptors'][:, :v1].transpose(0, 2, 1),
        'image1_shape': (1, 1, *img1_float.shape),
    }

    res = sg_jax(data, training=False)
    return {
        'kpts0': np.array(data['keypoints0'][0]),
        'kpts1': np.array(data['keypoints1'][0]),
        'matches0': np.array(res['matches0'][0]),
        'matching_scores0': np.array(res['matching_scores0'][0]),
    }

def run_pytorch_lg(img0_float, img1_float):
    with torch.no_grad():
        inp0 = torch.from_numpy(img0_float).unsqueeze(0).unsqueeze(0).to(device)
        inp1 = torch.from_numpy(img1_float).unsqueeze(0).unsqueeze(0).to(device)

        out0 = sp_torch({'image': inp0})
        out1 = sp_torch({'image': inp1})

        data = {
            'image0': {
                'keypoints': out0['keypoints'][0].unsqueeze(0),
                'descriptors': out0['descriptors'][0].unsqueeze(0),
                'image_size': torch.tensor([[inp0.shape[3], inp0.shape[2]]], device=device).float()
            },
            'image1': {
                'keypoints': out1['keypoints'][0].unsqueeze(0),
                'descriptors': out1['descriptors'][0].unsqueeze(0),
                'image_size': torch.tensor([[inp1.shape[3], inp1.shape[2]]], device=device).float()
            }
        }

        res = lg_torch(data)
        return {
            'kpts0': data['image0']['keypoints'][0].cpu().numpy(),
            'kpts1': data['image1']['keypoints'][0].cpu().numpy(),
            'matches0': res['matches0'][0].cpu().numpy(),
            'matching_scores0': res['matching_scores0'][0].cpu().numpy(),
        }

def visualize_matches(img0, img1, res, title):
    kpts0 = res['kpts0']
    kpts1 = res['kpts1']
    matches0 = res['matches0']

    h, w = img0.shape
    composite = np.zeros((h, w * 2 + 10), dtype=np.uint8)
    composite[:, :w] = img0
    composite[:, w+10:] = img1
    composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)

    valid_indices = np.where(matches0 > -1)[0]
    for idx in valid_indices:
        q_pos = (int(kpts0[idx][0]), int(kpts0[idx][1]))
        t_pos = (int(kpts1[matches0[idx]][0] + w + 10), int(kpts1[matches0[idx]][1]))
        cv2.line(composite, q_pos, t_pos, (0, 255, 0), 1)

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} - {len(valid_indices)} matches")
    plt.axis('off')
    plt.show()
