import torch
import os
import numpy as np
import openmesh as om


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


def seal(verts, faces, left=False):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (verts[circle_v_id, :]).mean(0)

    verts = np.vstack([verts, center])
    center_v_id = verts.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        if left:
            new_faces = [circle_v_id[i-1], center_v_id, circle_v_id[i]]
        else:
            new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id]
        faces = np.vstack([faces, new_faces])
    return verts, faces

