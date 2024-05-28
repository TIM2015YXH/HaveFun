
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import igl


def rotation_matrix_x(theta):
    """返回绕x轴旋转theta弧度的旋转矩阵"""
    R = np.array([[1,             0,              0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta),  np.cos(theta)]])
    return R

def rotation_matrix_y(theta):
    """返回绕y轴旋转theta弧度的旋转矩阵"""
    R = np.array([[ np.cos(theta), 0, np.sin(theta)],
                [             0, 1,             0],
                [-np.sin(theta), 0, np.cos(theta)]])
    return R

def rotation_matrix_z(theta):
    """返回绕z轴旋转theta弧度的旋转矩阵"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,             0,             1]])
    return R

def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)

def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)



def make_aligned(scale, transl):
    """make points vertical and scaled and translated, the orgin of smplx is same as the orgin of 
    joints:Nj x 3
    """
    make_vertical = rotation_matrix_z(-torch.pi/2) @ rotation_matrix_x(-torch.pi/2)
    make_scale = torch.eye(3)
    make_scale[:3, :3] *= scale
    aligned_matrix = torch.eye(4)
    aligned_matrix[:3,:3] = make_scale @ make_vertical
    aligned_matrix = aligned_matrix.to(joints.device)
    joints = (aligned_matrix @ to_homogeneous(joints).T).T[:,:3]
    
    # make_translated = torch.tensor([0,0.07,0]).reshape(3,1).to(set_ori_at_joint4.device)
    # aligned_matrix[:3,3:] = make_translated - set_ori_at_joint4
    aligned_matrix[:3,3:] = - transl
    return aligned_matrix
    # return (make_vertical @ verts.T).T + make_translated

def get_dmtet_weights(pts, verts, faces, lbs_weights_rest):
    
    """
    input:
        pts: [N, 3], dmtet results mesh verts
        verts: [7231, 3], aligned rest pose handy mesh vertices
        faces: [14368, 3], handy mesh faces
        lbs_weights_rest: [7231, 16], lbs weights of handy hand in rest pose
    output:
        dm_lbs_weights: [N, 16], lbs_weight of each vertex of dmtet result mesh
    """
    
    assert len(pts.shape) == 2, 'pts should have shape [num_samples, 3]'
    assert pts.shape[-1] == 3
    
    device, dtype = lbs_weights_rest.device, lbs_weights_rest.dtype
    
    num_samples, _ = pts.shape
    pts = pts.detach().cpu().numpy()
    verts = verts.detach().cpu().numpy().astype(np.float32)
    lbs_weights_rest = lbs_weights_rest.detach().cpu().numpy()
    
    dist2, f_id, closest = igl.point_mesh_squared_distance(pts, verts, faces[:, :3]) # [N], [N], [N,3]

    chosen_faces = faces[:, :3][f_id] # [N, 3] points_index of each face 

    closest_tri = verts[chosen_faces]
    barycentric = igl.barycentric_coordinates_tri(closest, closest_tri[:, 0, :].copy(), closest_tri[:, 1, :].copy(), closest_tri[:, 2, :].copy())
    dm_lbs_weights = (lbs_weights_rest[chosen_faces] * barycentric[..., None]).sum(axis=1)
    
    dm_lbs_weights = to_tensor(dm_lbs_weights ,dtype, device)
    
    return dm_lbs_weights
    
    
    
class MANOHand:
    n_keypoints = 21

    n_joints = 21

    center = 4

    root = 0

    labels = [
        'W', #0
        'I0', 'I1', 'I2', #3
        'M0', 'M1', 'M2', #6
        'L0', 'L1', 'L2', #9
        'R0', 'R1', 'R2', #12
        'T0', 'T1', 'T2', #15
        'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
    ]

    # finger tips are not keypoints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

    parents = [
        None,
        0, 1, 2,
        0, 4, 5,
        0, 7, 8,
        0, 10, 11,
        0, 13, 14,
        3, 6, 9, 12, 15
    ]

    end_points = [0, 16, 17, 18, 19, 20]



            