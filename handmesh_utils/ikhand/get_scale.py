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
import copy
from smplx import build_layer
import argparse


def batch_rodrigues(rot_vecs, epsilon: float = 1e-8):
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
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
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

mano_path = '/cpfs/shared/public/chenxingyu/models/mano/MANO_RIGHT.pkl'
smpl_cfgs = {
            'model_folder': mano_path,
            'model_type': 'mano',
            'num_betas': 10
        }
smpl_model = build_layer(
            smpl_cfgs['model_folder'], model_type = smpl_cfgs['model_type'],#the model_type is mano for DART dataset
            num_betas = smpl_cfgs['num_betas']
        )


device = 'cpu'


mano_size = math_np.measure_hand_size(
    utils.load_official_mano_model(mano_path)['keypoints_mean'],
    skeletons.MANOHand
) * config.MANO_SCALE # joints: (21, 3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_root', type=str, required=False, default='few_shot_data/real_hand/8717/shape_crop', help="png root")
    parser.add_argument('--front_id', type=str, required=False, default='000082')
    parser.add_argument('--back_id', type=str, required=False, default='000260')
    parser.add_argument('--dst_path', type=str, required=False, default='xx', help="imgs save root")
    
    opt = parser.parse_args()

    shape_root = opt.shape_root
    front_id = opt.front_id
    back_id = opt.back_id
    dst_path = f"../{opt.dst_path}"

    path1 = f'{shape_root}/{front_id}/ckpts/param.pth'
    path2 = f'{shape_root}/{back_id}/ckpts/param.pth'

    _, pose1 = torch.load(path1, map_location=torch.device('cpu'))
    _, pose2 = torch.load(path2, map_location=torch.device('cpu'))

    pose1 = pose1[0].cpu()
    pose2 = pose2[0].cpu()

    # sh, _ = torch.load(path1, map_location=torch.device('cpu'))
    # shape1 = sh

    shape1, _ = torch.load(path1, map_location=torch.device('cpu'))
    shape2, _ = torch.load(path2, map_location=torch.device('cpu'))


    theta_rodrigues1 = batch_rodrigues(pose1.reshape(-1, 3)).reshape(1, 16, 3, 3)
    theta_rodrigues2 = batch_rodrigues(pose2.reshape(-1, 3)).reshape(1, 16, 3, 3)

    __theta1 = theta_rodrigues1.reshape(1, 16, 3, 3)
    __theta2 = theta_rodrigues2.reshape(1, 16, 3, 3)

    pose1 = pose1.unsqueeze(0).to(shape1.device)
    pose2 = pose2.unsqueeze(0).to(shape1.device)


    so1 = smpl_model(betas = shape1.cpu(), hand_pose = __theta1[:, 1:].float(), global_orient = __theta1[:, 0].view(1, 1, 3, 3).float())
    smpl_v1 = so1['vertices'].clone().reshape(-1, 3).cpu().numpy()
    joints1 = so1['joints'].clone().reshape(-1, 3).cpu().numpy()

    so2 = smpl_model(betas = shape2.cpu(), hand_pose = __theta2[:, 1:].float(), global_orient = __theta2[:, 0].view(1, 1, 3, 3).float())
    smpl_v2 = so2['vertices'].clone().reshape(-1, 3).cpu().numpy()
    joints2 = so2['joints'].clone().reshape(-1, 3).cpu().numpy()



    fingertip1 = np.array([smpl_v1[333], smpl_v1[444], smpl_v1[672], smpl_v1[555], smpl_v1[744]]).reshape(5,3)
    alljionts1 = np.concatenate((joints1, fingertip1), axis=0)

    fingertip2 = np.array([smpl_v2[333], smpl_v2[444], smpl_v2[672], smpl_v2[555], smpl_v2[744]]).reshape(5,3)
    alljionts2 = np.concatenate((joints2, fingertip2), axis=0)

    print(alljionts1.shape)
    print(alljionts2.shape)

    real_size1 = math_np.measure_hand_size(
        alljionts1,
        skeletons.MANOHand
    ) * config.MANO_SCALE

    real_size2 = math_np.measure_hand_size(
        alljionts2,
        skeletons.MANOHand
    ) * config.MANO_SCALE

    scale1 = real_size1 / mano_size
    scale2 = real_size2 / mano_size
    print(scale1)
    print(scale2)
    os.makedirs(dst_path, exist_ok=True)
    np.save(os.path.join(dst_path, f"real_scale_{front_id}.npy"), scale1)
    np.save(os.path.join(dst_path, f"real_scale_{back_id}.npy"), scale2)

    np.save(os.path.join(dst_path, f"real_scale_{front_id}{back_id}.npy"), [scale1, scale2])
