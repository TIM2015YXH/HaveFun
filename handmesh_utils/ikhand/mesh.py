import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ikhand.math_np as math_np
import ikhand.utils as utils
import numpy as np


class MeshModel():
  def __init__(self, model_path):
    params = utils.load_official_mano_model(model_path)
    self.parents = params['parents']
    self.weights = params['weights']
    self.faces = params['faces']
    self.J_regressor = params['keypoint_regressor']
    self.shape_basis = params['shape_basis']
    self.mesh = params['mesh']
    self.ones = np.ones([params['mesh'].shape[0], 1])
    self.n_joints = len(self.parents)

  def set_params(self, abs_rotmat, beta=None):
    verts = self.mesh
    if beta is not None:
      verts = verts + np.einsum('C, VDC -> VD', beta, self.shape_basis)
    keypoints = np.einsum('VD, JV -> JD', verts, self.J_regressor)
    bones = math_np.keypoints_to_bones_batch([keypoints], self.parents)[0]
    posed_keypoints, _ = \
      math_np.forward_kinematics_batch([bones], [abs_rotmat], self.parents)
    posed_keypoints = posed_keypoints[0]
    J = posed_keypoints - np.einsum('JHW, JW -> JH', abs_rotmat, keypoints)
    G = np.concatenate([abs_rotmat, np.expand_dims(J, -1)], -1)
    verts = np.concatenate([verts, self.ones], 1)
    posed_verts = np.einsum(
      'VJ, JVD -> VD', self.weights, np.einsum('JHW, VW -> JVH', G, verts)
    )
    posed_keypoints = np.dot(self.J_regressor, posed_verts)
    return posed_keypoints, posed_verts

  def set_params_batch(self, abs_rotmat, beta):
    verts = self.mesh + np.einsum('NC, VDC -> NVD', beta, self.shape_basis)
    keypoints = np.einsum('NVD, JV -> NJD', verts, self.J_regressor)
    bones = math_np.keypoints_to_bones_batch(keypoints, self.parents)
    posed_keypoints, _ = \
      math_np.forward_kinematics_batch(bones, abs_rotmat, self.parents)
    J = posed_keypoints - np.einsum('NJHW, NJW -> NJH', abs_rotmat, keypoints)
    G = np.concatenate([abs_rotmat, np.expand_dims(J, -1)], -1)
    verts = np.concatenate([verts, np.ones(verts.shape[:-1] + (1,))], -1)
    posed_verts = np.einsum(
      'VJ, NJVD -> NVD', self.weights, np.einsum('NJHW, NVW -> NJVH', G, verts)
    )
    return posed_verts
