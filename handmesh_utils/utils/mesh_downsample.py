import sys
sys.path.insert(0, '/Users/chenxingyu/Documents/hand_mesh')
import os.path as osp
from utils import mesh_sampling, utils
from psbody.mesh import Mesh
import pickle
from read import save_mesh
import numpy as np


def downsample(transform_fp, template_fp, ds_factors):
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

    return tmp


if __name__ == '__main__':
    template_fp = osp.join('..', 'template', 'template.ply')
    transform_fp = osp.join('..', 'template', 'transform200.pkl')
    tmp = downsample(transform_fp, template_fp, [3,])
    ori_v = tmp['vertices'][0]
    down_matrix = tmp['down_transform'][-1].tocoo()
    down_index = down_matrix.col
    new_v = ori_v[down_index]
    J = np.load('../data/PanoHand/J_reg.npy')

    save_mesh('test.obj', new_v, tmp['face'][-1])
