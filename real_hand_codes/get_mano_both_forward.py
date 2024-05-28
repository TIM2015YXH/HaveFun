import torch
import numpy as np
import trimesh
import json
import os
from smplx import build_layer
import sys
sys.path.append('dmdrive')
from model.arbitary_conversion import matrix_to_axis_angle
import copy
import argparse

rot_y = torch.tensor([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]).float()

smpl_cfgs = {
            'model_folder': '/cpfs/shared/public/chenxingyu/models/mano/MANO_RIGHT.pkl',
            'model_type': 'mano',
            'num_betas': 10
        }
smpl_model = build_layer(
            smpl_cfgs['model_folder'], model_type = smpl_cfgs['model_type'],#the model_type is mano for DART dataset
            num_betas = smpl_cfgs['num_betas']
        )

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_root', type=str, required=False, default='few_shot_data/real_hand/8717/shape_crop', help="shape root")
    parser.add_argument('--front_id', type=str, required=False, default='000082')
    parser.add_argument('--back_id', type=str, required=False, default='000260')
    parser.add_argument('--dst_path', type=str, required=False, default='zzzzz_test_realhand', help="save params root")
    
    opt = parser.parse_args()
    
    path1 = os.path.join(opt.shape_root, f'{opt.front_id}/ckpts/param.pth')
    path2 = os.path.join(opt.shape_root, f'{opt.back_id}/ckpts/param.pth')

    dst_path = opt.dst_path
    os.makedirs(dst_path, exist_ok=True)

    _, pose1 = torch.load(path1, map_location=torch.device('cpu'))
    _, pose2 = torch.load(path2, map_location=torch.device('cpu'))
    # print(pose1)
    # print(pose2)
    pose1 = pose1[0].cpu()
    pose2 = pose2[0].cpu()


    sh, _ = torch.load(path1, map_location=torch.device('cpu'))
    shape1 = shape2 = sh

    # print(shape1, shape2)
    torch.save((shape1, pose1.unsqueeze(0).to(shape1.device)), os.path.join(dst_path, f'{opt.front_id}_mano_param.pth'))
    torch.save((shape2, pose2.unsqueeze(0).to(shape1.device)), os.path.join(dst_path, f'{opt.back_id}_mano_param.pth'))



    theta_rodrigues1 = batch_rodrigues(pose1.reshape(-1, 3)).reshape(1, 16, 3, 3)
    theta_rodrigues2 = batch_rodrigues(pose2.reshape(-1, 3)).reshape(1, 16, 3, 3)
    __theta1 = theta_rodrigues1.reshape(1, 16, 3, 3)
    __theta2 = theta_rodrigues2.reshape(1, 16, 3, 3)

    __theta1rot_y = copy.deepcopy(__theta1)
    __theta2rot_y = copy.deepcopy(__theta2)
    __theta1rot_y[:, 0] = torch.bmm(rot_y.unsqueeze(0), __theta1[:, 0])
    __theta2rot_y[:, 0] = torch.bmm(rot_y.unsqueeze(0), __theta2[:, 0])

    glo1 = matrix_to_axis_angle(__theta1rot_y[:, 0]).to(pose1.dtype).to(shape1.device)
    glo2 = matrix_to_axis_angle(__theta2rot_y[:, 0]).to(pose2.dtype).to(shape1.device)
    pose1 = pose1.unsqueeze(0).to(shape1.device)
    pose2 = pose2.unsqueeze(0).to(shape1.device)

    pose1[:, :3] = glo1.reshape(1,3)
    pose2[:, :3] = glo2.reshape(1,3)

    torch.save((shape1, pose1), os.path.join(dst_path, f'{opt.front_id}_flip_mano_param.pth'))
    torch.save((shape2, pose2), os.path.join(dst_path, f'{opt.back_id}_flip_mano_param.pth'))
    # print(shape1.shape, pose2.shape)

    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    so1 = smpl_model(betas = shape1.cpu(), hand_pose = __theta1[:, 1:].float(), global_orient = __theta1[:, 0].view(1, 1, 3, 3).float())
    so2 = smpl_model(betas = shape2.cpu(), hand_pose = __theta2[:, 1:].float(), global_orient = __theta2[:, 0].view(1, 1, 3, 3).float())
    smpl_v1 = so1['vertices'].clone().reshape(-1, 3).cpu().numpy()
    joints1 = so1['joints'].clone().reshape(-1, 3).cpu().numpy()
    smpl_v2 = so2['vertices'].clone().reshape(-1, 3).cpu().numpy()
    joints2 = so2['joints'].clone().reshape(-1, 3).cpu().numpy()

    f = smpl_model.faces

    shape_scale = np.load(os.path.join(dst_path, f"real_scale_{opt.front_id}.npy"))

    scale = 5 / shape_scale

    smpl_v1 *= scale
    joints1 *= scale
    smpl_v2 *= scale
    joints2 *= scale

    torch.save(sh, os.path.join(dst_path, f"shape_{opt.front_id}.pth"))



    m1 = trimesh.Trimesh(smpl_v1 - joints1[4], f, process=False)
    m2 = trimesh.Trimesh(smpl_v2 - joints2[4], f, process=False)


    angle_z = np.pi
    angle_y = np.pi
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z, [0, 0, 1])
    rotation_matrix_y = trimesh.transformations.rotation_matrix(angle_y, [0, 1, 0])
        
    m1.apply_transform(rotation_matrix_z)
    # m1.apply_transform(rotation_matrix_y)
    m2.apply_transform(rotation_matrix_z)
    # m2.apply_transform(rotation_matrix_y)


    m1_flip = copy.deepcopy(m1)
    m1_flip.apply_transform(rotation_matrix_y)
    m2_flip = copy.deepcopy(m2)
    m2_flip.apply_transform(rotation_matrix_y)


    translation_vector = [0, 0.07, 0]

    m1.apply_translation(translation_vector)
    m2.apply_translation(translation_vector)
    m1_flip.apply_translation(translation_vector)
    m2_flip.apply_translation(translation_vector)

    v_779m1 = 0
    v_779m2 = 0
    v_779m1_flip = 0
    v_779m2_flip = 0
    wrist_ids = [121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120, 108, 79, 78]
    for w in wrist_ids:
        v_779m1 += m1.vertices[w]
        v_779m2 += m2.vertices[w]
        v_779m1_flip += m1_flip.vertices[w]
        v_779m2_flip += m2_flip.vertices[w]
        
    v_779m1 /= len(wrist_ids)
    v_779m2 /= len(wrist_ids)
    v_779m1_flip /= len(wrist_ids)
    v_779m2_flip /= len(wrist_ids)
    _v_779m1 = np.round(v_779m1, 8)
    _v_779m2 = np.round(v_779m2, 8)
    _v_779m1_flip = np.round(v_779m1_flip, 8)
    _v_779m2_flip = np.round(v_779m2_flip, 8)
    print('m1:', _v_779m1)
    print('m2:', _v_779m2)
    print('m1_flip:', _v_779m1_flip)
    print('m2_flip:', _v_779m2_flip)



    m1.export(os.path.join(dst_path, f'{opt.front_id}.obj'))
    m2.export(os.path.join(dst_path, f'{opt.back_id}.obj'))
    m1_flip.export(os.path.join(dst_path, f'{opt.front_id}_flip.obj'))
    m2_flip.export(os.path.join(dst_path, f'{opt.back_id}_flip.obj'))


    rot_z180 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    rot_y180 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])

    aligned_scaled_joints1 = (rot_z180 @ (joints1 - joints1[4]).T).T
    aligned_scaled_joints2 = (rot_z180 @ (joints2 - joints2[4]).T).T
    aligned_scaled_joints1_flip = (rot_y180 @ rot_z180 @ (joints1 - joints1[4]).T).T
    aligned_scaled_joints2_flip = (rot_y180 @ rot_z180 @ (joints2 - joints2[4]).T).T

    np.save(os.path.join(dst_path, f'x5-4_joints{opt.front_id}.npy'), aligned_scaled_joints1)
    np.save(os.path.join(dst_path, f'x5-4_joints{opt.back_id}.npy'), aligned_scaled_joints2)
    np.save(os.path.join(dst_path, f'x5-4_joints{opt.front_id}_flip.npy'), aligned_scaled_joints1_flip)
    np.save(os.path.join(dst_path, f'x5-4_joints{opt.back_id}_flip.npy'), aligned_scaled_joints2_flip)
