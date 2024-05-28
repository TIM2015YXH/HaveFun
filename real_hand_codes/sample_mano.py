import smplx
from smplx import build_layer
import torch
import numpy as np
import argparse
from tqdm import trange
import os
import copy
import re
from pytorch3d.io import load_obj

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
    
def process_mesh(filename, tmp_mesh):
    
    with open(filename, "r") as ff:
        vertex = ff.readlines()

        vertex_lst = []
        for idx, vvv in enumerate(vertex):
            if idx >= 779:
                break
            else:
                vertex_lst.append(vvv.strip())
    
    with open(tmp_mesh, "r") as file:
        content = file.readlines()

        template_lst = []
        for item in content:
            if "usemtl Default_OBJ" in item or '#' in item or 'pytorch3d' in item or 'mtllib' in item:
                continue
            elif 'vn' in item:
                continue
            elif 'f' in item:
                pattern = r'\/([^\/\s]+)(?=\s|$)'
                matches = re.findall(pattern, item)
                item = re.sub(pattern, '', item)
            template_lst.append(item.strip())

    return template_lst, vertex_lst




parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, required=True, help='numbers of how many meshes to sample')
parser.add_argument('--dst_path', type=str, required=True, help='save path of all hand meshes')
parser.add_argument('--scale', type=float, required=True, help='scale of normal distribution')
parser.add_argument('--real_hand_scale', type=str, required=True, help='scale of normal distribution')
parser.add_argument('--font_id', type=int, required=True, help='font id')
parser.add_argument('--shape_root', type=str, required=True, help="shape root")

arg = parser.parse_args()

smpl_cfgs = {
            'model_folder': '/cpfs/shared/public/chenxingyu/models/mano/MANO_RIGHT.pkl',
            'model_type': 'mano',
            'num_betas': 10
        }
smpl_model = build_layer(
            smpl_cfgs['model_folder'], model_type = smpl_cfgs['model_type'],#the model_type is mano for DART dataset
            num_betas = smpl_cfgs['num_betas']
        )


dst_path = arg.dst_path

real_hand_scale = np.load(arg.real_hand_scale)
font_id = str(arg.font_id).zfill(6)

for i in trange(arg.num):
    sub_path = os.path.join(dst_path, str(i))
    os.makedirs(sub_path, exist_ok=True)
    print(sub_path)
    
    # poses = np.load('/cpfs/user/wangshaohui/stable-dreamfusion/dmdrive/pose_data/test_poseseq.npy')[0].reshape(48)
    poses = np.zeros(48)
    poses = torch.from_numpy(poses)

    # shape = torch.randn((1, 10)).float() * arg.scale
    print(f'{arg.shape_root}/{font_id}/ckpts/param.pth')
    shape = torch.load(f'{arg.shape_root}/{font_id}/ckpts/param.pth')[0].to(poses.device)
    # np.save(os.path.join(sub_path, f'shape_{i}.npy'), shape)

    

    theta_rodrigues = batch_rodrigues(poses.reshape(-1, 3)).reshape(1, 16, 3, 3)
    __theta = theta_rodrigues.reshape(1, 16, 3, 3)
    so = smpl_model(betas = shape, hand_pose = __theta[:, 1:].float(), global_orient = __theta[:, 0].view(1, 1, 3, 3).float())
    print(shape)
    smpl_v = so['vertices'].clone().reshape(-1, 3).cpu().numpy() / real_hand_scale
    joints = so['joints'].clone().reshape(-1, 3).cpu().numpy() / real_hand_scale

    # set origin at joint 4
    smpl_v -= joints[4]

    import trimesh

    mesh = trimesh.Trimesh(vertices=smpl_v, faces=smpl_model.faces, process=False).export(os.path.join(sub_path, f'{i}.obj'))
    
    v_779 = 0
    wrist_ids = [121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120, 108, 79, 78]
    for w in wrist_ids:
        v_779 += smpl_v[w]
        
    v_779 /= len(wrist_ids)
    _v779 = np.round(v_779, 8)

    np.save(os.path.join(sub_path, 'v_779.npy'), v_779)
    
    template_lst, vertex_lst = process_mesh(os.path.join(sub_path, f'{i}.obj'), 'real_hand_codes/0906_template_mano_close.obj')

    ccc = copy.deepcopy(template_lst)
    content_vtvn = vertex_lst + ['v {:.8f} {:.8f} {:.8f}\n'.format(_v779[0], _v779[1], _v779[2])] + ccc[779:]

    for idx in range(len(content_vtvn)):
        content_vtvn[idx] = content_vtvn[idx] + "\n"

    with open(os.path.join(sub_path,"779points.obj"), "w") as file:
        file.write(content_vtvn[0])
        file.write("mtllib material_0.mtl"+ "\n")
        file.write("usemtl material_0"+ "\n")
        file.writelines(content_vtvn[1:])
        # file.writelines(content_vtvn)
        
  
    mesh_scale = trimesh.load(os.path.join(sub_path,"779points.obj"), process=False, force='mesh')
  
    scale_factor = 5

    mesh_scale.apply_scale(scale_factor)
    angle_x = -np.pi/2  
    angle_z = -np.pi/2  
    rotation_matrix_x = trimesh.transformations.rotation_matrix(angle_x, [1, 0, 0])
    rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_z, [0, 0, 1])
    mesh_scale.apply_transform(rotation_matrix_x)
    mesh_scale.apply_transform(rotation_matrix_z)
    
    translation_vector = [0., 0.07, 0.]

    mesh_scale.apply_translation(translation_vector)
    mesh_scale.export(os.path.join(sub_path,f"779points_scaled_{i}.obj"))
    
    
    
    
    
    
    
    
    
    
    
    
    
    