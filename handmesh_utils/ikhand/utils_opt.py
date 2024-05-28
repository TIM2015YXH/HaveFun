import pickle
import numpy as np
import ikhand.skeletons as sk
from vctoolkit import math_np
import matplotlib.pyplot as plt


def load_official_mano_model(data_path, side=None):
    if side is None:
        for s in ['left', 'right']:
            if s in data_path.lower():
                side = s
        if side is None:
            raise RuntimeError('unable to resolve handedness from ' + data_path)
    assert side in ['left', 'right']

    with open(data_path, 'rb') as f:
        mano = pickle.load(f, encoding='latin1')

    mano['shapedirs'] = np.array(mano['shapedirs'])
    if side == 'left':
        mano['shapedirs'][:, 0, :] *= -1

    mesh = mano['v_template']

    kpts_mean = np.empty([sk.MANOHand.n_keypoints, 3], dtype=np.float32)
    kpts_mean[:16] = mano['J']
    for k, v in sk.MANOHand.mesh_mapping.items():
        kpts_mean[k] = mesh[v]

    J_regressor = np.zeros([21, mano['J_regressor'].shape[1]])
    J_regressor[:16] = mano['J_regressor'].toarray()
    for k, v in sk.MANOHand.mesh_mapping.items():
        J_regressor[k, v] = 1
    kpts_std = np.einsum('VDC, JV -> CJD', mano['shapedirs'], J_regressor)

    rel_axangle = mano['hands_mean'] + \
                  np.einsum('HW, WD -> HD', mano['hands_coeffs'], mano['hands_components'])
    rel_axangle = np.reshape(rel_axangle, [-1, 15, 3])

    # translate pose and skinning weight: we use child joint while MANO uses parent
    rel_axangle = \
        np.concatenate([np.zeros([rel_axangle.shape[0], 1, 3]), rel_axangle], 1)
    new_pose = np.zeros([rel_axangle.shape[0], sk.MANOHand.n_keypoints, 3], dtype=np.float32)
    new_weights = np.zeros([mano['weights'].shape[0], sk.MANOHand.n_keypoints], dtype=np.float32)
    for finger in 'TIMRL':
        new_pose[:, sk.MANOHand.labels.index(finger + '0')] = \
            rel_axangle[:, sk.MANOHand.labels.index('W')]
        new_weights[:, sk.MANOHand.labels.index(finger + '0')] = \
            mano['weights'][:, sk.MANOHand.labels.index('W')] / 5
        for joint in [1, 2, 3]:
            tar_idx = sk.MANOHand.labels.index(finger + str(joint))
            src_idx = sk.MANOHand.labels.index(finger + str(joint - 1))
            new_pose[:, tar_idx] = rel_axangle[:, src_idx]
            new_weights[:, tar_idx] = mano['weights'][:, src_idx]

    pack = {
        'kpts_mean': kpts_mean,
        'kpts_std': kpts_std,
        'rel_axangle': new_pose,
        'mesh': mesh,
        'weights': new_weights,
        'parents': sk.MANOHand.parents,
        'faces': mano['f'],
        'keypoint_regressor': J_regressor,
        'shape_basis': mano['shapedirs']
    }
    return pack


def centralize(kpts, center_idx, batch=False):
    if not batch:
        kpts = np.expand_dims(kpts, 0)
    x = kpts - kpts[:, center_idx:center_idx+1]
    if not batch:
        x = x[0]
    return x


def convert_skeleton_batch(data, src, tar):
    data = np.array(data)
    return np.stack([data[:, src.index(l)] for l in tar], 1)


def left_right_flip(v, f):
    v[:, -1] *= -1
    f = np.flip(f, axis=-1).copy()
    return v, f


def get_view_matrix(side):
    # hard-code view matrix for visualization rendering
    views = ['right', 'back', 'left', 'front']

    assert side in views

    view_mat = np.dot(
        math_np.convert(np.array([0, 0, np.pi/2]), 'axangle', 'rotmat'),
        math_np.convert(np.array([np.pi/2, 0, 0]), 'axangle', 'rotmat')
    )

    angle = views.index(side) * np.pi / 2
    view_mat = np.dot(
        math_np.convert(np.array([0, angle, 0]), 'axangle', 'rotmat'), view_mat
    )

    return view_mat


def extend_original_mano_pose(pose):
    # translate pose: we use child joint while MANO uses parent
    new_pose = np.zeros([sk.MANOHand.n_keypoints, 3], dtype=np.float32)
    for finger in 'TIMRL':
        new_pose[sk.MANOHand.labels.index(finger + '0')] = \
            pose[sk.MANOHand.labels.index('W')]
        for joint in [1, 2, 3]:
            tar_idx = sk.MANOHand.labels.index(finger + str(joint))
            src_idx = sk.MANOHand.labels.index(finger + str(joint - 1))
            new_pose[tar_idx] = pose[src_idx]
    return new_pose


def measure_hand_size(kpts, skeleton):
    bones = math_np.keypoints_to_bones(kpts, skeleton.parents)
    bones = np.array(
        [[bones[skeleton.labels.index(f + k)] for f in 'IMR'] for k in '0123']
    )
    return np.mean(np.linalg.norm(bones, axis=-1))
