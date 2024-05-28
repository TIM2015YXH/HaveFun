from vctoolkit import math_np
import ikhand.utils_opt as utils_opt
import numpy as np


class MeshModel():
    def __init__(self, model_path):
        params = utils_opt.load_official_mano_model(model_path)
        self.parents = params['parents']
        self.weights = params['weights']
        self.faces = params['faces']
        self.j_regressor = params['keypoint_regressor']
        self.shape_basis = params['shape_basis']
        self.mesh = params['mesh']
        self.ones = np.ones([params['mesh'].shape[0], 1])
        self.n_kpts = len(self.parents)

    def set_params(self, abs_rotmat, shape, scale=1.0, batch=False, center_idx=None):
        if not batch:
            abs_rotmat = np.expand_dims(abs_rotmat, 0)
            shape = np.expand_dims(shape, 0)

        verts = self.mesh + np.einsum('nc, vdc -> nvd', shape, self.shape_basis)
        kpts = np.einsum('nvd, jv -> njd', verts, self.j_regressor)
        bones = math_np.keypoints_to_bones(kpts, self.parents, batch=True)
        posed_kpts, _ = \
            math_np.forward_kinematics(bones, abs_rotmat, self.parents, batch=True)
        j = posed_kpts - np.einsum('njhw, njw -> njh', abs_rotmat, kpts)
        g = np.concatenate([abs_rotmat, np.expand_dims(j, -1)], -1)
        verts = np.concatenate([verts, np.ones(verts.shape[:-1] + (1,))], -1)
        posed_verts = np.einsum(
            'vj, njvd -> nvd', self.weights, np.einsum('njhw, nvw -> njvh', g, verts)
        )
        if center_idx is not None:
            posed_verts = posed_verts - posed_kpts[:, center_idx:center_idx + 1]
            posed_kpts = posed_kpts - posed_kpts[:, center_idx:center_idx + 1]

        if not batch:
            posed_kpts = posed_kpts[0]
            posed_verts = posed_verts[0]

        return posed_kpts * scale, posed_verts * scale
