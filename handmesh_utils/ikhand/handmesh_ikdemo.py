import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ikhand import math_np, utils, skeletons, config, network, mesh, optimize
import torch
import pickle
import numpy as np
import vctoolkit as vc
import cv2
from manotorch.manolayer import ManoLayer

device = 'cpu'
mano_path = "../template/MANO_RIGHT.pkl"
ikhand_model_path = '/Users/chenxingyu/Datasets/models/10600.pth'
mesh_path = '/Users/chenxingyu/Datasets/hand_test/mesh/wrist_test/IMG_5108.MOV/handmesh.pkl'
ik_path = '/Users/chenxingyu/Datasets/hand_test/ik/wrist_test/IMG_5108.MOV/ikmesh.json'
os.makedirs(os.path.dirname(ik_path), exist_ok=True)

camera_f = 1493#1662.768
camera_c = 512#[1080/2, 1920/2]

official_mano = utils.load_official_mano_model(mano_path)
mano_size = math_np.measure_hand_size(
    official_mano['keypoints_mean'],
    skeletons.MANOHand
) * config.MANO_SCALE

j_reg = official_mano['keypoint_regressor']
# load IK model
# model = network.IKNetV1(
#     skeletons.KWAIHand.n_keypoints * 3,
#     skeletons.MANOHand.n_joints * 6,
#     shape_dim=10, depth=4, width=1024
# ).to(device)

model = network.IKNetConvV1(
    778 * 3,
    skeletons.MANOHand.n_joints * 6,
    shape_dim=10, depth=4, width=1024
).to(device)

model.load_state_dict(torch.load(ikhand_model_path, map_location=torch.device(device))['model'])
model.eval()

# MANO file
mano_layer = ManoLayer(
    rot_mode='axisang',
    use_pca=False,
    side='right',
    center_idx=0,
    mano_assets_root=os.path.join(os.path.dirname(__file__), '../template'),
    flat_hand_mean=True,
)

# read pose file
with open(mesh_path, 'rb') as f2:
    pose_dict_test = pickle.load(f2)

# solve camera RT
pnp_solver = optimize.PnPSolver(camera_f, camera_c)

save_dict = {}
for sample in vc.progress_bar( pose_dict_test.items() ):
    image_path = sample[0]
    root_ori = sample[1]['root']
    verts_ori = sample[1]['verts']
    xyz = np.dot(j_reg, verts_ori)
    verts = verts_ori - xyz[4:5]
    uv_ori =sample[1]['pose2d'].astype(np.float32)
    scale = mano_size / math_np.measure_hand_size(xyz, skeletons.MANOHand)
    verts *= scale
    verts[:, 0] *= -1
    with torch.no_grad():
        mano_params = model(torch.from_numpy(np.reshape(verts, [1, -1])).float().to(device))

    mano_params = {k: v.detach().cpu().numpy() for k, v in mano_params.items()}
    mano_params['pose'] = np.reshape(mano_params['pose'], [skeletons.MANOHand.n_keypoints, 6])
    mano_params['shape'] = np.reshape(mano_params['shape'], [-1])
    mano_params['scale'] = mano_params['scale'][0][0]

    abs_rotmat = math_np.convert(mano_params['pose'], 'rot6d', 'rotmat')
    # from left hand to right hand
    mirror = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    abs_rotmat = np.einsum('nhw, wd -> nhd', abs_rotmat, mirror)
    rel_rotmat = math_np.rotmat_abs_to_rel(abs_rotmat, skeletons.MANOHand.parents)
    rel_axangle = math_np.convert(rel_rotmat, 'rotmat', 'axangle')
    mano_params['pose_ori'] = utils.mano_pose_21_to_16(rel_axangle)
    manopth_ret = mano_layer(
        torch.tensor(mano_params['pose_ori'].copy(), dtype=torch.float32).view(1, -1),
        torch.zeros(1, 10, dtype=torch.float32)
    )

    mano_kpts = manopth_ret.joints[0].cpu().numpy() * mano_params['scale']
    mano_verts = manopth_ret.verts[0].cpu().numpy() * mano_params['scale']

    camera_r, camera_t, _, mano_kpts_proj = pnp_solver.solve(mano_kpts, uv_ori)
    save_dict.update({image_path: {'root_ori': root_ori.tolist(), 'verts_ori': verts_ori.tolist(), '3d': xyz.tolist(), '2d': uv_ori.tolist(),
                                   'camera_r':camera_r.tolist(), 'camera_t': camera_t.tolist(),
                                   'mano_verts': mano_verts.tolist(), 'mano_kpts': mano_kpts.tolist(),
                                   'theta': mano_params['pose_ori'].tolist(), 'beta': mano_params['shape'].tolist(), 'scale': float(mano_params['scale']),
                                   'rel_scale': float(scale)
                                   }
                      })

vc.save(ik_path, save_dict)
