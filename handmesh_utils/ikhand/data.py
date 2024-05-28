import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import math_np
import ikhand.utils as utils
import skeletons as sk
import config_xchen as cfg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mesh
import vctoolkit as vc


# TODO: grouped priority sampling for higher efficiency


class MANODatasetINTRP(Dataset):
  def __init__(self, settings, data_path=cfg.MANO_PATH):
    self.shape_std = settings['shape_std']
    self.scale_std = settings['scale_std']
    self.noise_std = settings['noise_std']
    self.conf_std = settings['conf_std']
    self.conf_mean = settings['conf_mean']
    self.batch_size = settings['batch_size']
    self.fingers = 'TIMRL'

    data = utils.load_official_mano_model(data_path)

    self.keypoints_mean = \
      data['keypoints_mean'].astype(np.float32) * cfg.MANO_SCALE
    self.keypoints_std = \
      data['keypoints_std'].astype(np.float32) * cfg.MANO_SCALE
    rel_axangle = data['rel_axangle'].astype(np.float32)

    rel_quat = math_np.convert(rel_axangle, 'axangle', 'quat')
    self.finger_poses = []
    for finger in self.fingers:
      finger_poses = []
      for joint in '0123':
        finger_poses.append(
          rel_quat[:, sk.MANOHand.labels.index(finger + joint)]
        )
      finger_poses = np.stack(finger_poses, 1)
      self.finger_poses.append(finger_poses)

    if settings['mix_fingers']:
      pool = np.concatenate([self.finger_poses[i] for i in range(1, 5)])
      self.finger_poses = [self.finger_poses[0], pool, pool, pool, pool]

    if settings['p_power'] != 0:
      self.indices = []
      self.priority = []
      for i in range(5):
        finger = self.finger_poses[i]
        angles = np.sum(np.arccos(finger[..., 0]) * 2, axis=-1)
        angles = np.power(angles, settings['p_power'])
        self.priority.append(angles / np.sum(angles, axis=0))
        self.indices.append(np.arange(finger.shape[0]))
    else:
      self.indices = \
        [np.arange(self.finger_poses[i].shape[0]) for i in range(5)]
      self.priority = [None] * 5

  def __sample_pose__(self):
    pose = []
    for i in range(5):
      idx = np.random.choice(
        self.indices[i], size=self.batch_size, p=self.priority[i]
      )
      pose.append(self.finger_poses[i][idx])
    pose = np.stack(pose, 1)
    pose = np.reshape(pose, [self.batch_size, -1, 4])
    return pose

  def __assemble_fingers__(self, quat):
    pose = \
      np.zeros([self.batch_size, sk.MANOHand.n_joints, 4], dtype=np.float32)
    pose[:, sk.MANOHand.root, 0] = 1 # set wrist to zero rotation
    for i, finger in enumerate(self.fingers):
      for j, knuckle in enumerate('0123'):
        pose[:, sk.MANOHand.labels.index(finger + knuckle)] = quat[:, i, j]
    return pose

  def __len__(self):
    return 1000

  def __getitem__(self, _):
    rel_quat_a = self.__sample_pose__()
    rel_quat_b = self.__sample_pose__()
    alpha = np.random.uniform(size=[self.batch_size, 1, 1]).astype(np.float32)
    rel_quat = math_np.slerp_batch(rel_quat_a, rel_quat_b, alpha)

    rel_quat = self.__assemble_fingers__(np.reshape(rel_quat, [-1, 5, 4, 4]))
    rel_rotmat = math_np.convert(rel_quat, 'quat', 'rotmat')
    rel_rotmat[:, sk.MANOHand.root] = math_np.random_rotation(self.batch_size)
    abs_rotmat = \
      math_np.rotmat_rel_to_abs_batch(rel_rotmat, sk.MANOHand.parents)
    abs_6d = math_np.convert(abs_rotmat, 'rotmat', 'rot6d')

    shape = np.random.normal(
      scale=self.shape_std, size=[self.batch_size, self.keypoints_std.shape[0]]
    ).astype(np.float32)
    scale = np.random.normal(
      loc=1.0, scale=self.scale_std, size=[self.batch_size, 1, 1]
    ).astype(np.float32)
    ref_keypoints = self.keypoints_mean + \
                    np.einsum('NC, CJD -> NJD', shape, self.keypoints_std)
    ref_keypoints *= scale

    ref_bones = math_np.keypoints_to_bones_batch(
      ref_keypoints, sk.MANOHand.parents
    )
    keypoints, _ = math_np.forward_kinematics_batch(
      ref_bones, abs_rotmat, sk.MANOHand.parents
    )

    conf = np.random.normal(
      size=[self.batch_size, keypoints.shape[1], 1],
      loc=self.conf_mean, scale=self.conf_std
    ).astype(np.float32)
    conf = np.clip(conf, 0, 1)
    noise = np.random.normal(
      size=keypoints.shape, loc=0,
      scale=self.noise_std * (1 - conf) + self.noise_std / 5
    ).astype(np.float32)
    input_keypoints = keypoints + noise

    input_keypoints = \
      utils.centralize(input_keypoints, sk.MANOHand.center, batch=True)
    inputs = np.reshape(
      np.concatenate([input_keypoints, conf], -1), [self.batch_size, -1]
    )
    keypoints = utils.centralize(keypoints, sk.MANOHand.center, batch=True)

    batch = {
      'inputs': inputs.astype(np.float32),
      'keypoints': keypoints.astype(np.float32),
      'ref_bones': ref_bones.astype(np.float32),
      'pose': np.reshape(abs_6d.astype(np.float32), [self.batch_size, -1]),
      'shape': shape.astype(np.float32),
      'scale': scale,
      'rel_quat': rel_quat
    }

    return batch


class MANODatasetOriginal(Dataset):
  def __init__(self, data_path=cfg.MANO_PATH):
    data = utils.load_official_mano_model(data_path)
    rel_axangle = data['rel_axangle'].astype(np.float32)
    self.batch_size = rel_axangle.shape[0]
    rel_rotmat = math_np.convert(rel_axangle, 'axangle', 'rotmat')
    abs_rotmat = math_np.rotmat_rel_to_abs_batch(rel_rotmat, sk.MANOHand.parents)
    ref_keypoints = data['keypoints_mean'].astype(np.float32) * cfg.MANO_SCALE
    ref_bones = math_np.keypoints_to_bones_batch([ref_keypoints], sk.MANOHand.parents)
    ref_bones = np.tile(ref_bones, [self.batch_size, 1, 1])
    keypoints, _ = math_np.forward_kinematics_batch(ref_bones, abs_rotmat, sk.MANOHand.parents)
    input_keypoints = utils.centralize(keypoints, sk.MANOHand.center, batch=True)
    inputs = np.reshape(
      np.concatenate([input_keypoints, np.ones(input_keypoints.shape[:2] + (1,))], -1),
      [self.batch_size, -1]
    )
    keypoints = utils.centralize(keypoints, sk.MANOHand.center, batch=True)
    self.batch = {
      'inputs': inputs.astype(np.float32),
      'keypoints': keypoints.astype(np.float32),
      'ref_bones': ref_bones.astype(np.float32),
      'pose': math_np.convert(abs_rotmat, 'rotmat', 'rot6d').astype(np.float32),
      'shape': np.zeros([inputs.shape[0], 10], dtype=np.float32),
      'scale': np.ones([inputs.shape[0], 1, 1], dtype=np.float32),
    }

  def __len__(self):
    return 1

  def __getitem__(self, i):
    return {k: v.copy() for k, v in self.batch.items()}


class MANODatasetKwaiWrapper(Dataset):
  # Load data which is compatible with Kwai's hand model that contains 20 joints
  # and without confidence prediction
  def __init__(self, core):
    self.core = core

  def __len__(self):
    return self.core.__len__()

  def __getitem__(self, idx):
    data = self.core[idx]
    inputs = np.reshape(
      data['inputs'],
      [self.core.batch_size, sk.MANOHand.n_keypoints, 4]
    )[:, :, :3]
    inputs = utils.convert_skeleton_batch(
      inputs, sk.MANOHand.labels, sk.KWAIHand.labels
    )
    data['inputs'] = np.reshape(inputs, [self.core.batch_size, -1])
    return data


class MANODatasetAllVertWrapper(Dataset):
  # The input being all 768 verts
  def __init__(self, core, noise_std=0.01):
    self.core = core
    self.mesh = mesh.MeshModel(cfg.MANO_PATH)
    self.noise_std = noise_std

  def __len__(self):
    return self.core.__len__()

  def __getitem__(self, idx):
    data = self.core[idx]
    abs_rotmat = math_np.convert(
      np.reshape(data['pose'], [-1, 6]), 'rot6d', 'rotmat'
    )
    abs_rotmat = np.reshape(abs_rotmat, [-1, sk.MANOHand.n_joints, 3, 3])
    verts = self.mesh.set_params_batch(abs_rotmat, data['shape'])
    verts = verts.astype(np.float32) * data['scale'] * cfg.MANO_SCALE
    noise = np.random.normal(size=verts.shape, scale=self.noise_std)
    verts += noise
    # TODO: smooth the mesh
    data['inputs'] = np.reshape(verts, [self.core.batch_size, -1])
    return data


class MANODatasetTipsWrapper(Dataset):
  # The input being the end points and the bone lengths
  def __init__(self, core):
    self.core = core
    self.core.noise_std = 0
    self.core.conf_std = 0
    self.core.conf_mean = 1
    self.mesh = mesh.MeshModel(cfg.MANO_PATH)

  def __len__(self):
    return self.core.__len__()

  def __getitem__(self, idx):
    data = self.core[idx]
    N = data['inputs'].shape[0]
    end_points = np.reshape(
      data['inputs'], [N, sk.MANOHand.n_keypoints, -1]
    )[:, sk.MANOHand.end_points, :3]
    bone_lengths = np.linalg.norm(data['ref_bones'], axis=-1)
    data['inputs'] = np.concatenate(
      [np.reshape(end_points, [N, -1]), np.reshape(bone_lengths, [N, -1])], -1
    )
    return data


class CombinedDataset(Dataset):
  def __init__(self, datasets):
    self.datasets = []
    self.partition = []
    self.volume = 0
    for ds, rep in datasets:
      for _ in range(rep):
        self.partition.append((self.volume, len(ds) + self.volume))
        self.datasets.append(ds)
        self.volume += len(ds)

  def __len__(self):
    return self.volume

  def __getitem__(self, idx):
    for ds, r in zip(self.datasets, self.partition):
      if idx >= r[0] and idx < r[1]:
        return ds[idx - r[0]]


class InfiniteDataLoader(DataLoader):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dataset_iterator = super().__iter__()

  def __iter__(self):
    return self

  def __next__(self):
    try:
      batch = next(self.dataset_iterator)
    except StopIteration:
      # Dataset exhausted, use a new fresh iterator.
      self.dataset_iterator = super().__iter__()
      batch = next(self.dataset_iterator)
    return batch
