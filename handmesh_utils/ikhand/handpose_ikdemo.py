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
import json

device = 'cpu'
mano_path = "../template/MANO_RIGHT.pkl"
ikhand_model_path = '/Users/chenxingyu/Datasets/models/26000.pth'
pose_path = '/Users/chenxingyu/Datasets/hand_test/pose/wrist_test/IMG_5108.MOV/handpose.pkl'
ik_path = '/Users/chenxingyu/Datasets/hand_test/ik/wrist_test/IMG_5108.MOV/ikpose_conv_oriz.json'
os.makedirs(os.path.dirname(ik_path), exist_ok=True)

camera_f = 1493
camera_c = 512

show = False

mano_size = math_np.measure_hand_size(
    utils.load_official_mano_model(mano_path)['keypoints_mean'],
    skeletons.MANOHand
) * config.MANO_SCALE

# load IK model
# model = network.IKNetV1(
#     skeletons.KWAIHand.n_keypoints * 3,
#     skeletons.MANOHand.n_joints * 6,
#     shape_dim=10, depth=4, width=1024
# ).to(device)

model = network.IKNetConvV1(
    skeletons.KWAIHand.n_keypoints * 3,
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
with open(pose_path, 'rb') as f2:
    pose_dict_test = pickle.load(f2)

# solve camera RT
pnp_solver = optimize.PnPSolver(camera_f, camera_c)

save_dict = {}
for sample in vc.progress_bar( pose_dict_test.items() ):
    image_path = sample[0]
    xyz_ori = sample[1]['3d']
    xyz = utils.centralize(xyz_ori, skeletons.KWAIHand.center)
    uv_ori =sample[1]['2d']
    uv =  np.array(uv_ori).astype(np.float32)
    scale = mano_size / math_np.measure_hand_size(xyz, skeletons.KWAIHand)
    xyz *= scale
    with torch.no_grad():
        mano_params = model(torch.from_numpy(np.reshape(xyz, [1, -1])).float().to(device))

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

    mano_kpts = manopth_ret.joints[0].cpu().numpy() * mano_params['scale'] / scale
    mano_verts = manopth_ret.verts[0].cpu().numpy() * mano_params['scale'] / scale
    mano_kpts_20 = utils.convert_skeleton_batch(
        [mano_kpts],
        skeletons.MPIIHand.labels, skeletons.KWAIHand.labels
    )[0]
    camera_r, camera_t, _, mano_kpts_20_proj = pnp_solver.solve(mano_kpts_20, uv)
    save_dict.update({image_path: {'3d': xyz_ori, '2d': uv_ori, 'camera_r':camera_r.tolist(), 'camera_t': camera_t.tolist(), 'xyz_input': np.reshape(xyz, [1, -1]).tolist(),
                                   'mano_verts': mano_verts.tolist(), 'mano_kpts': mano_kpts.tolist(), 'theta6d': mano_params['pose'].tolist(),
                                   'theta': mano_params['pose_ori'].tolist(), 'beta': mano_params['shape'].tolist(), 'scale': float(mano_params['scale']),
                                   'rel_scale': float(scale), 'face': mano_layer.th_faces.numpy().tolist(), 'camera_f': camera_f, 'camera_c': camera_c
                                   }
                      })
    #//////
    # rotmat = math_np.convert(mano_params['pose'], 'rot6d', 'rotmat')
    # # from left hand to right hand
    # mirror = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # rotmat = np.einsum('nhw, wd -> nhd', rotmat, mirror)
    # axang = math_np.convert(rotmat, 'rotmat', 'axangle')
    # # apply the parameters
    # mano_mesh = mesh.MeshModel(mano_path)
    # mano_kpts, mano_verts = mano_mesh.set_params(rotmat, mano_params['shape'])
    # mano_kpts = mano_kpts * mano_params['scale'] / scale
    # mano_verts = mano_verts * mano_params['scale'] / scale
    #
    # # convert mano 21 keypoints to KWAI 20 keypoints
    # mano_kpts_20 = utils.convert_skeleton_batch([mano_kpts], skeletons.MANOHand.labels, skeletons.KWAIHand.labels)[0]
    # camera_r, camera_t, _, mano_kpts_proj = pnp_solver.solve(mano_kpts_20, np.flip(uv, axis=-1).copy())
    # mano_params['pose_ori'] = utils.recover_original_mano_pose(mano_params['pose'])
    # save_dict.update({image_path: {'3d': xyz_ori, '2d': uv_ori, 'camera_r':camera_r.tolist(), 'camera_t': camera_t.tolist(),
    #                                'mano_verts': mano_verts.tolist(), 'mano_kpts': mano_kpts.tolist(),
    #                                'theta': mano_params['pose_ori'].tolist(), 'beta': mano_params['shape'].tolist(), 'scale': mano_params['scale'],
    #                                'rel_scale': scale
    #                                }
    #                   })
    # if show:
    #     canvas = vc.load(image_path)
    #     a = vc.render_bones_from_uv(
    #         np.flip(uv, axis=-1).copy(), canvas.copy(),
    #         skeletons.KWAIHand.parents, skeletons.KWAIHand.colors
    #     )
    #     cv2.putText(a, "hand pose", (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)
    #     b = vc.render_bones_from_uv(
    #         mano_kpts_proj, canvas.copy(),
    #         skeletons.KWAIHand.parents, skeletons.KWAIHand.colors
    #     )
    #     cv2.putText(b, "ikhand", (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)
    #     frame = np.concatenate([a, b], 1)
    #     vc.save('keypoints.jpg', frame)
    #     mano_verts_global = np.einsum('hw, vw -> vh', camera_r, mano_verts) + np.reshape(camera_t, [1, 3])
    #     vc.save('mesh.obj', (mano_verts_global, mano_mesh.faces))

with open(ik_path, 'w') as fo:
    json.dump(save_dict, fo)
