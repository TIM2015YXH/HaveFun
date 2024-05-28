import pickle
import numpy as np
import ikhand.skeletons as sk
import os
import ikhand.config as cfg
import vctoolkit as vc
import torch
import ikhand.math_np as math_np


def load_official_mano_model(data_path):
  with open(data_path, 'rb') as f:
    mano = pickle.load(f, encoding='latin1')

  mesh = mano['v_template']

  keypoints_mean = np.empty([sk.MANOHand.n_keypoints, 3], dtype=np.float32)
  keypoints_mean[:16] = mano['J']
  for k, v in sk.MANOHand.mesh_mapping.items():
    keypoints_mean[k] = mesh[v]

  J_regressor = np.zeros([21, mano['J_regressor'].shape[1]])
  J_regressor[:16] = mano['J_regressor'].toarray()
  for k, v in sk.MANOHand.mesh_mapping.items():
    J_regressor[k, v] = 1
  keypoints_std = np.einsum('VDC, JV -> CJD', mano['shapedirs'], J_regressor)

  rel_axangle = mano['hands_mean'] + \
    np.einsum('HW, WD -> HD', mano['hands_coeffs'], mano['hands_components'])
  rel_axangle = np.reshape(rel_axangle, [-1, 15, 3])

  # translate pose and skinning weight: we use child joint while MANO uses parent
  rel_axangle = \
    np.concatenate([np.zeros([rel_axangle.shape[0], 1, 3]), rel_axangle], 1)
  new_pose = np.zeros([rel_axangle.shape[0], sk.MANOHand.n_joints, 3], dtype=np.float32)
  new_weights = np.zeros([mano['weights'].shape[0], sk.MANOHand.n_joints], dtype=np.float32)
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
    'keypoints_mean': keypoints_mean,
    'keypoints_std': keypoints_std,
    'rel_axangle': new_pose,
    'mesh': mesh,
    'weights': new_weights,
    'parents': sk.MANOHand.parents,
    'faces': mano['f'],
    'keypoint_regressor': J_regressor,
    'shape_basis': mano['shapedirs']
  }
  return pack


def centralize(keypoints, center_idx, batch=False):
  if not batch:
    keypoints = np.expand_dims(keypoints, 0)
  x = keypoints - keypoints[:, center_idx:center_idx+1]
  if not batch:
    x = x[0]
  return x


def get_latest_checkpoint(model_name):
  file_path = os.path.join(cfg.CKPT_DIR, model_name, cfg.LATEST_MODEL_INFO_FILE)
  if os.path.isfile(file_path):
    return vc.load_json(file_path)
  else:
    return None


def change_learning_rate(optimizer, learning_rate):
  for g in optimizer.param_groups:
    g['lr'] = learning_rate


def convert_skeleton_batch(data, src, tar):
  data = np.array(data)
  return np.stack([data[:, src.index(l)] for l in tar], 1)


def left_right_flip(v, f):
  v[:, -1] *= -1
  f = np.flip(f, axis=-1).copy()
  return v, f


def restore_state(model_name, model, optimizer, scheduler, settings):
  step = 0
  latest_ckpt = get_latest_checkpoint(model_name)
  if latest_ckpt is not None:
    ckpt = torch.load(latest_ckpt['path'])
    missing_keys, _ = model.load_state_dict(ckpt['model'], strict=False)
    if not missing_keys:
      try:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
      except Exception as e:
        print(e)
      step = ckpt['step']
    else:
      print('Partially loaded model. Missing keys:')
      for k in missing_keys:
        print(k)
    print(f'Loaded model at {step}')
  else:
    print('Start a new model.')

  return step


def collect_batch_input(iterations, datasets):
  batch_input = {}
  for k in iterations.keys():
    try:
      pack = next(iterations[k])
    except StopIteration:
      iterations[k] = iter(datasets[k]['dataloader'])
      pack = next(iterations[k])
    for k, v in pack.items():
      if k not in batch_input:
        batch_input[k] = []
      batch_input[k].append(v[0])
  batch_input = {k: torch.cat(v) for k, v in batch_input.items()}
  return batch_input


def wrap_to_batch(x, fn):
  return fn(np.expand_dims(x, 0))[0]


def recover_original_mano_pose(pose):
  pose = math_np.convert(np.reshape(pose, [-1, 6]), 'rot6d', 'rotmat')
  pose = math_np.rotmat_abs_to_rel(pose, sk.MANOHand.parents)
  pose = math_np.convert(pose, 'rotmat', 'axangle')
  mano_pose = np.zeros([16, 3])
  mano_pose[0] = pose[4] # use M0 to represent palm
  for finger in 'TIMRL':
    for knuckle in [0, 1, 2]:
      mano_pose[sk.MANOHand.labels.index(finger + str(knuckle))] = \
        pose[sk.MANOHand.labels.index(finger + str(knuckle + 1))]
  return mano_pose


def mano_pose_21_to_16(rel_pose):
  new_pose = np.zeros([16, 3])
  new_pose[0] = rel_pose[0].copy()
  for finger in 'TIMRL':
    for knuckle in [0, 1, 2]:
      new_pose[sk.MANOHand.labels.index(finger + str(knuckle))] = \
        rel_pose[sk.MANOHand.labels.index(finger + str(knuckle + 1))].copy()
  return new_pose


def mano_pose_16_to_21(rel_pose):
  if rel_pose.shape[0] == 48:
    rel_pose = rel_pose.reshape([16, 3])
  new_pose = np.zeros([21, 3])
  new_pose[0] = rel_pose[0].copy()
  for finger in 'TIMRL':
    for knuckle in [0, 1, 2, 3]:
      if knuckle == 0:
        new_pose[sk.MANOHand.labels.index(finger + str(knuckle))] = \
          np.zeros_like(rel_pose[sk.MANOHand.labels.index('W')].copy()) + 1e-5
      else:
        new_pose[sk.MANOHand.labels.index(finger + str(knuckle))] = \
          rel_pose[sk.MANOHand.labels.index(finger + str(knuckle - 1))].copy()
  return new_pose


def mano_pts_to_parents(joints):
  if joints.shape[0] == 63:
    joints = joints.reshape(21, 3)
  new_joints = np.zeros([21, 3])
  new_joints[0] = joints[0].copy()
  for finger in 'TIMRL':
    for knuckle in [0, 1, 2, 3]:
      bone_idx = sk.MANOHand.labels.index(finger + str(knuckle))
      parent_idx = sk.MANOHand.parents[bone_idx]
      new_joints[bone_idx] = joints[parent_idx].copy()
  return new_joints
