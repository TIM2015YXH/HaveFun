import torch
import openmesh as om
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import os


def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)


def save_obj(v, f, obj_ori, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for line in open(obj_ori, 'r'):
        if line.startswith('vt'):
            obj_file.write(line)
        if line.startswith('vn'):
            x, y, z = [float(i) for i in line.split(' ')[1:]]
            x *= -1
            new_line = 'vn ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
            obj_file.write(new_line)
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

dir = '/Users/chenxingyu/Datasets/Virtual_Hand/result/model/1-10/1-10_24'
save_path = dir.replace('model', 'flip_model')
file = '2.obj'
mesh = read_mesh(os.path.join(dir, file))
# mesh = read_mesh('../template/template.ply')
mesh.x[:, 0] *= -1
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_obj(mesh.x.numpy(), mesh.face.numpy().T, os.path.join(dir, file), os.path.join(save_path, file)) # '../template/template.obj')

