import torch
import numpy as np

sample_rate = 10
# path = "/workspace/songrise/CLIP-Actor/datasets/amass/SFU/SFU/0008/0008_Yoga001_poses.npz"
path = "Extended_1_stageii.npz"
with open(path, "rb") as f:
    data = np.load(f, allow_pickle=True)
    print(data.files)
    for k in data.files:
        print(k)
        print(data[k])

    print('poses', data['poses'].shape)
    print('pose_body', data['pose_body'].shape)
    print('pose_hand', data['pose_hand'].shape)
    print('pose_jaw', data['pose_jaw'].shape)
    print('pose_eye', data['pose_eye'].shape)
    poses = data["poses"]
    poses = poses[::sample_rate]
    betas = data["betas"][:10]
    poses = poses.reshape(-1, 165)
    # poses[:,0,:] = 0.0
    poses = poses.astype(np.float32)
    print(poses.shape)
    with open("Extended_1_stageii.npy", "wb") as f:
        np.save(f, poses)

    
print(f"Done with processing")