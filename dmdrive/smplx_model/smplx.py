# Code based on ROMP: https://github.com/Arthur151/ROMP/blob/master/romp/lib/models/smpl.py
# License from ROMP: https://github.com/Arthur151/ROMP/blob/master/LICENSE


# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(r.dtype).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R



class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class SMPLX(nn.Module):
    def __init__(self,
                 model_path,
                 gender='neutral',
                 device=None):
        ''' SMPL model constructor'''
        super(SMPLX, self).__init__()
        dtype = torch.float32
        self.dtype = dtype
        self.device = device
        # create the SMPLX model
        if osp.isdir(model_path):
            model_fn = 'SMPLX_{}.{ext}'.format(gender.upper(), ext='pkl')
            smplx_path = osp.join(model_path, model_fn)
        else:
            smplx_path = model_path
        assert osp.exists(smplx_path), 'Path {} does not exist!'.format(
            smplx_path)

        with open(smplx_path, 'rb') as smplx_file:
            data_struct = Struct(**pickle.load(smplx_file,
                                               encoding='latin1'))
        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long, device=device))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype, device=device))

        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(data_struct.shapedirs), dtype=dtype, device=device))

        shapedirs = data_struct.shapedirs

        expr_dirs = shapedirs[:, :, 300:310]
        self.register_buffer(
            'expr_dirs', to_tensor(to_np(expr_dirs), dtype=dtype))

        self.register_buffer('J_regressor',
                             to_tensor(to_np(data_struct.J_regressor), dtype=dtype, device=device))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        # print(posedirs.shape)
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype, device=device))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0]), device=device).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(data_struct.weights), dtype=dtype, device=device))

        left_hand_mean = data_struct.hands_meanl

        right_hand_mean = data_struct.hands_meanr

        self.register_buffer('left_hand_mean',
                             to_tensor(left_hand_mean, dtype=self.dtype))
        self.register_buffer('right_hand_mean',
                             to_tensor(right_hand_mean, dtype=self.dtype))
        
        pose_mean = self.create_mean_pose(data_struct)
        # pose_mean_tensor = pose_mean.clone().to(dtype)
        # pose_mean_tensor = torch.tensor(pose_mean, dtype=dtype)
        self.register_buffer('pose_mean', to_tensor(to_np(pose_mean), dtype=self.dtype))

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([21 * 3],
                                     dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = np.concatenate([global_orient_mean, body_pose_mean,
                                    jaw_pose_mean,
                                    leye_pose_mean, reye_pose_mean,
                                    self.left_hand_mean, self.right_hand_mean],
                                   axis=0)

        return pose_mean

    def verts_transformations(self,
                              poses,
                              betas,
                              expression = None,
                              transl=None,
                              return_tensor=True,
                              concat_joints=False,
                              return_delta_v=False):
        ''' Forward pass for SMPLX model but also return transformation of each vertex

        Parameters
        ----------
        betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_tensor: bool, optional
            Return in torch.tensor. (default=True)
        concat_joints: bool, optional
            Concat joints info at the end. (default=False)
        '''
        bn = poses.shape[0]
        assert bn == 1
        if isinstance(poses, np.ndarray):
            poses = to_tensor(poses, self.dtype, self.device)
        if isinstance(betas, np.ndarray):
            betas = to_tensor(betas, self.dtype, self.device)
        if isinstance(transl, np.ndarray):
            transl = to_tensor(transl, self.dtype, self.device)
        full_poses = poses + self.pose_mean
        if expression is not None:
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        L, vertices, delta_v = lbs(shape_components, full_poses, self.v_template,
                          shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          return_T=True, concat_joints=concat_joints)
        if transl is not None:
            transl_4x4 = torch.eye(4, dtype=self.dtype, device=self.device)[None]
            transl_4x4[0, :3, 3] = transl.unsqueeze(1)
            T = torch.matmul(transl_4x4, L)
        else:
            T = L
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()[0]
            T = T.detach().cpu().numpy()[0]
        return vertices, T, delta_v

    def verts_transformations_avatar(self,
                              poses,
                              v_shaped,
                              transl=None,
                              return_tensor=True,
                              concat_joints=False):
        ''' Forward pass for SMPLX model but also return transformation of each vertex
            this version does not use the beta of SMPLX, but directly shaped verticies
        Parameters
        ----------
        betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_tensor: bool, optional
            Return in torch.tensor. (default=True)
        concat_joints: bool, optional
            Concat joints info at the end. (default=False)
        '''
        bn = poses.shape[0]
        assert bn == 1
        if isinstance(poses, np.ndarray):
            poses = to_tensor(poses, self.dtype, self.device)
        if isinstance(v_shaped, np.ndarray):
            v_shaped = to_tensor(v_shaped, self.dtype, self.device)
        if isinstance(transl, np.ndarray):
            transl = to_tensor(transl, self.dtype, self.device)
        full_poses = poses + self.pose_mean
        L, vertices = my_lbs(v_shaped, full_poses, self.v_template,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, 
                          return_T=True, concat_joints=concat_joints)
        if transl is not None:
            transl_4x4 = torch.eye(4, dtype=self.dtype, device=self.device)[None]
            transl_4x4[0, :3, 3] = transl.unsqueeze(1)
            T = torch.matmul(transl_4x4, L)
        else:
            T = L
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()[0]
            T = T.detach().cpu().numpy()[0]
        return vertices, T
    

    # def forward_v_shaped(self,
    #                     poses,
    #                     v_shaped,
    #                     transl=None,
    #                     return_joints=False,
    #                     return_tensor=True):
    #     ''' Forward pass for SMPLX model, but take shaped vertices as input
    #     '''
    #     bn = poses.shape[0]
    #     assert bn == 1
    #     if isinstance(poses, np.ndarray):
    #         poses = to_tensor(poses, self.dtype, self.device)
    #     if isinstance(v_shaped, np.ndarray):
    #         v_shaped = to_tensor(v_shaped, self.dtype, self.device)
    #     if isinstance(transl, np.ndarray):
    #         transl = to_tensor(transl, self.dtype, self.device)
    #     vertices, joints = my_lbs(v_shaped, poses, self.v_template,
    #                            self.shapedirs, self.posedirs,
    #                            self.J_regressor, self.parents,
    #                            self.lbs_weights)
    #     if transl is not None:
    #         vertices = vertices + transl.unsqueeze(1)
    #         joints = joints + transl.unsqueeze(1)
    #     if not return_tensor:
    #         vertices = vertices.detach().cpu().numpy()
    #         joints = joints.detach().cpu().numpy()
    #     if return_joints:
    #         return vertices[0], joints[0]
    #     return vertices[0]

    def forward(self,
                poses,
                betas,
                expression=None,
                transl=None,
                return_joints=False,
                return_tensor=True):
        ''' Forward pass for SMPLX model

        Parameters
        ----------
        betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_joints: bool, optional
            Return (vertices, joints) tuple. (default=False)
        return_tensor: bool, optional
            Return in torch.tensor. (default=True)
        '''
        bn = poses.shape[0]
        assert bn == 1
        if isinstance(poses, np.ndarray):
            poses = to_tensor(poses, self.dtype, self.device)
        if isinstance(betas, np.ndarray):
            betas = to_tensor(betas, self.dtype, self.device)
        if isinstance(transl, np.ndarray):
            transl = to_tensor(transl, self.dtype, self.device)
        full_poses = poses + self.pose_mean
        if expression is not None:
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs
        vertices, joints = lbs(shape_components, full_poses, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)
        if transl is not None:
            vertices = vertices + transl.unsqueeze(1)
            joints = joints + transl.unsqueeze(1)
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
            joints = joints.detach().cpu().numpy()
        if return_joints:
            return vertices, joints
        return vertices


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).contiguous().view(
        batch_size, -1, 3)

    lmk_faces = lmk_faces + torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3).contiguous()[lmk_faces].contiguous().view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks



def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32, return_T=False, concat_joints=False):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_delta = blend_shapes(betas, shapedirs[...,:betas.shape[-1]])
    v_shaped = v_template + v_delta

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # print(J.shape)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        # print(rot_mats)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = v_shaped + pose_offsets

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    if return_T:
        if concat_joints:
            return torch.cat([T, A], dim=1), torch.cat([v_posed, J], dim=1), v_delta
        else:
            return T, v_posed, v_delta

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed

def my_lbs(v_shaped, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot: bool = True, return_T = False, concat_joints = False):
    #! Jan 11: this lbs does not require the beta of SMPL, this is used for skinning the avatarCLIP dataset
    batch_size = pose.shape[0]
    device, dtype = pose.device, pose.dtype

    # Add shape contribution
    # v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    if return_T:
        if concat_joints:
            return torch.cat([T, A], dim=1), torch.cat([v_posed, J], dim=1)
        else:
            return T, v_posed

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed

def smplx_lbs(betas, pose, v_template, shapedirs, posedirs, joints, parents, lbs_weights, dtype=torch.float32, pose2rot=True, expression=None, expr_dirs=None):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        joints: torch.tensor [NB, 3]
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    assert len(v_template.shape) == 3, 'pts should have shape [B, num_vertices, 3]'

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # # Add shape contribution
    # v_delta = blend_shapes(betas, shapedirs)
    # v_shaped = v_template + v_delta
    if expression is None:
        v_shaped = v_template
    else:
        v_shaped = v_template + blend_shapes(expression, expr_dirs)

    # Get the joints
    # NxJx3 array
    # J = vertices2joints(J_regressor, v_shaped)
    J = joints

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        # print(pose_feature.shape)
        rot_mats = pose.view(batch_size, -1, 3, 3)
        # print(posedirs.shape)

        # pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
        #                             posedirs).view(batch_size, -1, 3)

    v_posed = v_shaped 

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = joints.shape[1]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    # if return_T:
    #     if concat_joints:
    #         return torch.cat([T, A], dim=1), torch.cat([v_posed, J], dim=1)
    #     else:
    #         return T, v_posed

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return torch.cat([T, A], dim=1), verts, J_transformed

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.contiguous().view(-1, 3, 3),
        rel_joints.contiguous().view(-1, 3, 1)).contiguous().view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

def convert_between(gt_pose, dst_pose, dtype=torch.float32):
    # batch_size = gt_pose.shape[0]
    gt_rot_mats = batch_rodrigues(
        gt_pose.view(-1, 3), dtype=dtype).view([-1, 3, 3])

    dst_rot_mats = batch_rodrigues(
        dst_pose.view(-1, 3), dtype=dtype).view([-1, 3, 3])
    
    # return torch.einsum('ijk,ikl->ijl', dst_rot_mats, torch.inverse(gt_rot_mats))

    return torch.inverse(gt_rot_mats)
    

if __name__ == '__main__':
    device = torch.device('cpu')
    smplx = SMPLX(
        '../../smplx/',
        gender='male',
        device=torch.device('cpu')
    )
    import os
    import trimesh
    def save_mesh(vertices, faces, path):
        """
        save mesh to path in .ply format
        """

        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(os.path.join(path))
        print(f'saved mesh to {path}')

    DATA_DIR = "/cpfs/user/yangqianhe/stable-dreamfusion-xchen/data/raw/00016/train/Take4"

    smplx_DIR = os.path.join(DATA_DIR, "SMPLX/mesh-f00050_smplx.pkl")
    with open(smplx_DIR, 'rb') as f:
        smplx_param = pickle.load(f)
    
    betas =torch.tensor(smplx_param['betas'].reshape(-1,10))
    expression = torch.tensor(smplx_param['expression'].reshape(-1,10))
    jaw_pose = torch.tensor(smplx_param['jaw_pose'].reshape(-1,3))
    leye_pose = torch.tensor(smplx_param['leye_pose'].reshape(-1,3))
    reye_pose = torch.tensor(smplx_param['reye_pose'].reshape(-1,3))
    right_hand_pose = torch.tensor(smplx_param['right_hand_pose'].reshape(-1,45))
    left_hand_pose = torch.tensor(smplx_param['left_hand_pose'].reshape(-1,45))
    transl = torch.tensor(smplx_param['transl'].reshape(-1,3))
    body_pose = torch.tensor(smplx_param['body_pose'].reshape(-1,63))
    global_orient = torch.tensor(smplx_param['global_orient'].reshape(-1,3))

    # global_orient = torch.zeros((1,3))
    full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                               body_pose.reshape(-1, 21, 3),
                               jaw_pose.reshape(-1, 1, 3),
                               leye_pose.reshape(-1, 1, 3),
                               reye_pose.reshape(-1, 1, 3),
                               left_hand_pose.reshape(-1, 15, 3),
                               right_hand_pose.reshape(-1, 15, 3)],
                              dim=1).reshape(-1, 165)
    # pose_path = 'Extended_1_stageii.npy'
    # pose_data = np.load(pose_path)
    # dst_pose = pose_data[0,...].reshape(-1,165)
    # dst_pose[:,:3] = np.zeros((1,3))

    dst_pose = full_pose
    dst_pose[:,:3] = torch.zeros((1,3))
            
    full_pose = torch.zeros(1,165)
    # betas = torch.zeros(1, 10)
    # betas[0, 0] = -2.0
    # full_pose[0,:3] = torch.zeros((1,3))
    # shape_components = torch.cat([betas, expression], dim=-1)
    verts, joints = smplx(full_pose, betas, transl=None, return_tensor=True, return_joints=True)
    

    # R_global = rodrigues(global_orient.unsqueeze(0).numpy())
    # print(R_global)
    # verts = torch.bmm(verts.float() , torch.tensor(R_global).float() )
    # joints = torch.bmm(joints , torch.tensor(R_global).float()).type_as(verts) 

    scale = 0.5
    root = joints[0,0,:].reshape(1,3)
    print(root)
    v_rest = (verts - root)*scale #[N,3]

    # save_mesh(v_rest.detach().cpu().numpy()[0], smplx.faces,'smplx_zero_global_2.ply')
    



    
    from utils import get_dmtet_weights

    faces_rest = smplx.faces.astype(np.int32) # [f, 3]

    
    # save_mesh(v_rest.detach().cpu().numpy(), faces_rest,'v_rest.ply')

    lbs_weights_rest = smplx.lbs_weights
    dm_lbs_weights = get_dmtet_weights(v_rest[0], v_rest[0], faces_rest, lbs_weights_rest)


    T_handy_rest2dst, verts_deform, joints_deform = smplx_lbs(betas=betas, pose=dst_pose, 
                                v_template=verts, shapedirs=smplx.shapedirs,
                                posedirs=smplx.posedirs, joints=joints, 
                                parents=smplx.parents, lbs_weights=dm_lbs_weights, pose2rot=True)
                

    

    verts_dst, joints_dst = smplx(dst_pose, betas, transl=None, return_tensor=True, return_joints=True)
    verts_dst = (verts_dst - joints_dst[0,0,:])*scale
    print(joints_dst[0,0,:])
    verts_deform = (verts_deform - joints_dst[0,0,:])*scale
    save_mesh(verts_dst.detach().cpu().numpy()[0], smplx.faces,'smplx_zero_pose_frame50.ply')
    print(verts_deform.shape)
    save_mesh(verts_deform.detach().cpu().numpy()[0], smplx.faces,'smplx_deform_frame50.ply')


