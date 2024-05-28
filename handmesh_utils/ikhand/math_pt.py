import numpy as np
import torch
import ikhand.config as cfg


def convert(rot, src, tar):
  eps = torch.tensor(torch.finfo(torch.float32).eps)
  if src == 'rot6d':
    data_shape = rot.shape[:-1]
    rot = rot.view(-1, 6)
    if tar == 'rotmat':
      col0 = rot[:, 0:3] / \
          torch.maximum(torch.linalg.norm(rot[:, 0:3], dim=-1, keepdim=True), eps)
      col1 = rot[:, 3:6] - torch.sum((col0 * rot[:, 3:6]), dim=-1, keepdim=True) * col0
      col1 = col1 / torch.maximum(torch.linalg.norm(col1, dim=-1, keepdim=True), eps)
      col2 = torch.cross(col0, col1)
      rotmat = torch.stack([col0, col1, col2], -1)
      rotmat = torch.reshape(rotmat, data_shape + (3, 3))
      return rotmat
  if src == 'axangle':
    data_shape = rot.shape[:-1]
    rot = rot.view(-1, 3)
    if tar == 'rotmat':
      theta = torch.norm(rot, dim=-1, keepdim=True)
      c = torch.cos(theta)
      s = torch.sin(theta)
      t = 1 - c
      x, y, z = torch.split(rot / torch.maximum(theta, eps), 1, dim=-1)
      rotmat = torch.stack([
        t*x*x + c, t*x*y - z*s, t*x*z + y*s,
        t*x*y + z*s, t*y*y + c, t*y*z - x*s,
        t*x*z - y*s, t*y*z + x*s, t*z*z + c
      ], 1)
      rotmat = torch.reshape(rotmat, data_shape + (3, 3))
      return rotmat

  raise NotImplementedError(f'Unsupported conversion: from {src} to {tar}.')


def bones_to_keypoints(bones, parents):
  keypoints = []
  for c, p in enumerate(parents):
    if p is None:
      keypoints.append(bones[:, c])
    else:
      keypoints.append(bones[:, c] + keypoints[p])
  keypoints = torch.stack(keypoints, 1)
  return keypoints


def forward_kinematics(ref_bones, abs_rotmat, parents):
  bones = torch.einsum('njhw, njw -> njh', abs_rotmat, ref_bones)
  keypoints = bones_to_keypoints(bones, parents)
  return keypoints, bones


def rot6d_fk(rot6d, ref_bones, skeleton, eps):
  rotmat = convert(rot6d, 'rot6d', 'rotmat', eps)
  keypoints, _ = forward_kinematics(ref_bones, rotmat, skeleton.parents)
  keypoints -= keypoints[:, skeleton.center:skeleton.center+1].to(cfg.DEVICE)
  return keypoints
