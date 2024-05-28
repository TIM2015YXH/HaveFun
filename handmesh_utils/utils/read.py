import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om
from os import path as osp
from utils import utils, mesh_sampling
from psbody.mesh import Mesh
import pickle


def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)


def save_mesh(fp, x, f):
    om.write_mesh(fp, om.TriMesh(x, f))


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs, face_uvs):
    # obj
    file = open(mesh_path, 'w')
    name = mesh_path.split('/')[-1].split('.')[0]
    file.write(f'mtllib {name}.mtl\n')

    for v in verts:
        file.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))

    for vt in uvs:
        file.write('vt %.6f %.6f\n' % (vt[0], vt[1]))

    file.write('usemtl Material.002\n')
    file.write('s off\n')
    for i, (f, fu) in enumerate(zip(faces, face_uvs)):
        f_plus = f + 1
        fu_plus = fu + 1
        file.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (f_plus[0], fu_plus[0], i+1,
                                              f_plus[1], fu_plus[1], i+1,
                                              f_plus[2], fu_plus[2], i+1,))
    file.close()
    # mtl
    file = open(mesh_path.replace('.obj', '.mtl'), 'w')
    file.write('newmtl mesh\n')
    file.write('Ns 100\n')
    file.write('Ka 0.000000 0.000000 0.000000\n')
    file.write('Kd 1.000000 1.000000 1.000000\n')
    file.write('Ks 0.500000 0.500000 0.500000\n')
    file.write('Ni 1.000000\n')
    file.write('d 1.000000\n')
    file.write('illum 0\n')
    file.write(f'map_Kd {name}.png')
    file.close()




def spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation, writer=None, device='cpu'):
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        # ds_factors = [3.5, 3.5, 3.5, 3.5]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
            mesh, ds_factors)
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
         'up_transform': U
        }

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], seq_length[idx], tmp['vertices'][idx], dilation[idx])#.to(device)
        for idx in range(len(tmp['face']) - 1)
    ]

    down_transform_list = [
        utils.to_sparse(down_transform)#.to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform)#.to(device)
        for up_transform in tmp['up_transform']
    ]

    return spiral_indices_list, down_transform_list, up_transform_list, tmp


if __name__ == '__main__':
    mesh = read_mesh('../data/FreiHAND/template/template.obj')
    save_mesh('../data/FreiHAND/template/template.obj', mesh.x.numpy(), mesh.face.numpy().T)
