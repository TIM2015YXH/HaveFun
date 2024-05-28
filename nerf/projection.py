import os
import numpy as np
import trimesh
import pickle

import torch
import torch.nn.functional as F

import pytorch3d
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d import _C


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(mesh_path, device=device):
    verts, faces, aux = load_obj(mesh_path, device=device)
    faces_idx = faces.verts_idx.to(device)
    faces_t = faces.textures_idx.to(device)
    verts_uvs = aux[1].to(device)
    return verts, faces, faces_idx, verts_uvs, faces_t, device

class ProjectXYZToUVD():
    def __init__(self):
        temp_handy_obj_path = 'hand_model/temp_vertical_trans0_07.obj'
        # texture_path = 'hand_model/Right_Hand_Shape.pkl'
        # with open(texture_path, 'rb') as f:
        #     texture_model = pickle.load(f)
        print('Loading template handy model...\n')
        verts, faces, faces_idx, verts_uvs, faces_t, device = load_model(temp_handy_obj_path)
        self.device = device
        self.verts = verts
        self.faces_idx = faces_idx
        self.faces = faces
        self.verts_uvs = verts_uvs
        self.faces_t = faces_t
        self.xyzc = verts
        self.load_mesh_topology(verts, faces_idx)

    def load_mesh_topology(self, verts, faces_idx, cache_path='hand_model/cache'):

        if not os.path.exists(os.path.join(cache_path, 'faces_to_corres_edges.npy')):
            print('==> Computing mesh topology... ', cache_path)
            faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces = self._parse_mesh(verts, faces_idx)
            # save cache
            os.makedirs(cache_path)
            np.save(os.path.join(cache_path, 'faces_to_corres_edges.npy'), faces_to_corres_edges.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, 'edges_to_corres_faces.npy'), edges_to_corres_faces.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, 'verts_to_corres_faces.npy'), verts_to_corres_faces.to('cpu').detach().numpy().copy())
            print('==> Finished! Cache saved to: ', cache_path)

        else:
            print('==> Find pre-computed mesh topology! Loading cache from: ', cache_path)
            faces_to_corres_edges = torch.from_numpy(np.load(os.path.join(cache_path, 'faces_to_corres_edges.npy')))
            edges_to_corres_faces = torch.from_numpy(np.load(os.path.join(cache_path, 'edges_to_corres_faces.npy')))
            verts_to_corres_faces = torch.from_numpy(np.load(os.path.join(cache_path, 'verts_to_corres_faces.npy')))
        
        ## smpl: [13776, 3],[20664, 2],[6890, 9]
        self.faces_to_corres_edges = faces_to_corres_edges.long().to(self.device)  # torch.Size([14372, 3])
        self.edges_to_corres_faces = edges_to_corres_faces.long().to(self.device)  # torch.Size([21959, 2])
        self.verts_to_corres_faces = verts_to_corres_faces.long().to(self.device)  # torch.Size([7588, 9])

    # parsing mesh (e.g. adjacency of faces, verts, edges, etc.)
    def _parse_mesh(self, verts, faces_idx, N_repeat_edges=2, N_repeat_verts=9):

        meshes = Meshes(verts=[verts], faces=[faces_idx])
        print('parsing mesh topology...')

        # compute faces_to_corres_edges
        faces_to_corres_edges = meshes.faces_packed_to_edges_packed()  # (13776, 3)

        # compute edges_to_corres_faces
        edges_to_corres_faces = torch.full((len(meshes.edges_packed()), N_repeat_edges), -1.0).to(self.device)  # (20664, 2)
        for i in range(len(faces_to_corres_edges)):
            for e in faces_to_corres_edges[i]:
                idx = 0
                while idx < edges_to_corres_faces.shape[1]:
                    if edges_to_corres_faces[e][idx] < 0:
                        edges_to_corres_faces[e][idx] = i
                        break
                    else:
                        idx += 1

        # compute verts_to_corres_faces
        verts_to_corres_faces = torch.full((len(verts), N_repeat_verts), -1.0).to(self.device)  # (6890, 9)
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                idx = 0
                while idx < verts_to_corres_faces.shape[1]:
                    if verts_to_corres_faces[v][idx] < 0:
                        verts_to_corres_faces[v][idx] = i
                        break
                    else:
                        idx += 1
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                verts_to_corres_faces[v][verts_to_corres_faces[v] < 0] = verts_to_corres_faces[v][0].clone()

        return faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces


    @torch.no_grad()
    def xyz_to_uvd(self, points, verts, points_inside_mesh_approx=True, scaling_factor=1):

        # STEP 0: preparation
        verts = self.verts
        faces = self.faces_idx[None, ...].repeat(len(verts), 1, 1)
        meshes = Meshes(verts=verts*scaling_factor, faces=faces)
        pcls = Pointclouds(points=points*scaling_factor)
        # compute nearest faces
        _, idx = self.point_mesh_face_distance(meshes, pcls)

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        triangles = triangles_meshes[idx]  # [batch*65536, 3, 3]

        # STEP 1: Compute the nearest point on the mesh surface
        nearest, stats = self._parse_nearest_projection(triangles, pcls.points_packed())

        if points_inside_mesh_approx:
            sign_tensor = self._calculate_points_inside_meshes_normals(pcls.points_packed(), nearest, triangles, meshes.verts_normals_packed()[meshes.faces_packed()][idx])
        else:
            sign_tensor = self._calculate_points_inside_meshes(pcls.points_packed(), meshes.verts_packed())

        # STEP 2-6: Compute the final projection point (check self._revise_nearest() for details)
        dist = torch.norm(pcls.points_packed() - nearest, dim=1)
        nearest_new, dist, idx = self._revise_nearest(pcls.points_packed(), idx, meshes, sign_tensor, nearest, dist, stats)
        h = dist * sign_tensor

        triangles = triangles_meshes[idx]
        barycentric = self.points_to_barycentric(triangles, nearest_new)

        # bad case
        barycentric = torch.clamp(barycentric, min=0.)
        barycentric = barycentric / (torch.sum(barycentric, dim=1, keepdim=True) + 1e-12)

        # # local_coordinates
        # local_coordinates_meshes = self._calculate_local_coordinates_meshes(meshes.faces_normals_packed(), triangles_meshes)
        # local_coordinates = local_coordinates_meshes[idx]

        h = h.view(len(verts), -1, 1)
        barycentric = barycentric.view(len(verts), -1, 3)
        idx = idx.view(len(verts), -1)

        # revert scaling
        h = h / scaling_factor
        nearest_new = nearest_new / scaling_factor
        nearest = nearest / scaling_factor

        return h, barycentric, idx, nearest_new, nearest # , local_coordinates
    
    
    @torch.no_grad()
    def _revise_nearest(self,
                        points,
                        idx,
                        meshes,
                        inside,
                        nearest,
                        dist,
                        stats,
                        ):

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        faces_normals_meshes = meshes.faces_normals_packed()
        verts_normals_meshes = meshes.verts_normals_packed()[meshes.faces_packed()]

        bc_ca_ab = self.faces_to_corres_edges[idx]
        a_b_c = meshes.faces_packed()[idx]

        is_a, is_b, is_c = stats['is_a'], stats['is_b'], stats['is_c']
        is_bc, is_ac, is_ab = stats['is_bc'], stats['is_ac'], stats['is_ab']

        nearest_new, dist_new, idx_new = nearest.clone(), dist.clone(), idx.clone()

        def _revise(is_x, x_idx, x_type):

            points_is_x = points[is_x]
            inside_is_x = inside[is_x]
            if x_type == 'verts':
                verts_is_x = a_b_c[is_x][:, x_idx]
                corres_faces_is_x = self.verts_to_corres_faces[verts_is_x]
                N_repeat = 9  # maximum # of adjacent faces for verts
            elif x_type == 'edges':
                edges_is_x = bc_ca_ab[is_x][:, x_idx]
                corres_faces_is_x = self.edges_to_corres_faces[edges_is_x]
                N_repeat = 2  # maximum # of adjacent faces for edges
            else:
                raise ValueError('x_type should be verts or edges')

            # STEP 2: Find a set T of all triangles containing s~
            triangles_is_x = triangles_meshes[corres_faces_is_x]
            verts_normals_is_x = verts_normals_meshes[corres_faces_is_x]
            faces_normals_is_x = faces_normals_meshes[corres_faces_is_x]

            # STEP 3: Vertex normal alignment
            verts_normals_is_x_aligned = self._align_verts_normals(verts_normals_is_x, triangles_is_x, inside_is_x)

            # STEP 4: Check if inside control volume
            points_is_x_repeated = points_is_x.unsqueeze(1).repeat(1, N_repeat, 1)
            inside_control_volume, barycentric = \
                self._calculate_points_inside_target_volume(points_is_x_repeated, triangles_is_x, verts_normals_is_x_aligned, faces_normals_is_x, return_barycentric=True)  # (n', N_repeat):bool, (n', N_repeat, 3)
            barycentric = torch.clamp(barycentric, min=0.)
            barycentric = barycentric / (torch.sum(barycentric, dim=-1, keepdim=True) + 1e-12)

            # STEP 5: compute set of canditate surface points {s}
            surface_points_set = (barycentric[..., None] * triangles_is_x).sum(dim=2)
            surface_to_points_dist_set = torch.norm(points_is_x_repeated - surface_points_set, dim=2) + 1e10 * (1 - inside_control_volume)  # [n', N_repeat]
            _, idx_is_x = torch.min(surface_to_points_dist_set, dim=1)  # [n', ]

            # STEP 6: Choose the nearest point to x from {s} as the final projection point
            surface_points = surface_points_set[torch.arange(len(idx_is_x)), idx_is_x]  # [n', 3]
            surface_to_points_dist = surface_to_points_dist_set[torch.arange(len(idx_is_x)), idx_is_x]  # [n', ]
            faces_is_x = corres_faces_is_x[torch.arange(len(idx_is_x)), idx_is_x]

            # update
            nearest_new[is_x] = surface_points
            dist_new[is_x] = surface_to_points_dist
            idx_new[is_x] = faces_is_x

        # revise verts
        if torch.any(is_a): _revise(is_a, 0, 'verts')
        if torch.any(is_b): _revise(is_b, 1, 'verts')
        if torch.any(is_c): _revise(is_c, 2, 'verts')

        # revise edges
        if torch.any(is_bc): _revise(is_bc, 0, 'edges')
        if torch.any(is_ac): _revise(is_ac, 1, 'edges')
        if torch.any(is_ab): _revise(is_ab, 2, 'edges')

        return nearest_new, dist_new, idx_new



    @torch.no_grad()
    def point_to_face_coord_naive(self, pts, vts=None, faces=None):
        # pts: (B,P,3) point clouds in the 3D space
        # vts: (B,N,3) mesh vertices
        # faces: (B,T,3) face indices

        # return:
        # dists： (B,P) point to face distances
        # idxs: (B,P) indices of the closest faces
        # baries: (B,P,3) barycentric coordinates of the projections on the closest faces
        _DEFAULT_MIN_TRIANGLE_AREA = 5e-4
        if len(pts.shape) == 2:
            pts = pts.unsqueeze(0)
        vts = self.verts.unsqueeze(0)
        faces = self.faces_idx.unsqueeze(0)
        
        pcls = Pointclouds(points=pts)
        meshes = Meshes(verts=vts,faces=faces)

        if len(meshes) != len(pcls):
            raise ValueError("meshes and pointclouds must be equal sized batches")
        B = len(meshes)

        # packed representation for pointclouds
        points = pcls.points_packed()  # (B*P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_tris = meshes.num_faces_per_mesh().max().item()    

        # print(tris_first_idx)

        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            _DEFAULT_MIN_TRIANGLE_AREA,
        )
        selected_tris = self.faces_idx[idxs]
       
        selected_uvs = self.verts_uvs[selected_tris].reshape(B, -1, 3, 2) # (B, P, 3, 2)
  
        baries = self.get_barycentric(points,tris[idxs]) # (B*P,3)
        # baries = self.points_to_barycentric(tris[idxs], points)

        baries = torch.clamp(baries, min=0.)
        baries = baries / (torch.sum(baries, dim=1, keepdim=True) + 1e-12)
        
        # dists = dists.reshape(B,-1) # (B,P)
        
        baries = baries.reshape(B,-1,3) # (B,P,3)

        uv_query = torch.einsum('bpq,bpqr->bpr', baries, selected_uvs).squeeze()
        
        # compute nearest faces

        # triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        # triangles = triangles_meshes[idxs]  # [batch*65536, 3, 3]

        # STEP 1: Compute the nearest point on the mesh surface
        nearest, stats = self._parse_nearest_projection(tris[idxs], pcls.points_packed())
    
        sign_tensor = self._calculate_points_inside_meshes_normals(pcls.points_packed(), nearest, tris[idxs], meshes.verts_normals_packed()[meshes.faces_packed()][idxs])
        
        # dist = torch.norm(pcls.points_packed() - nearest, dim=1)
        # nearest_new, dist, idx = self._revise_nearest(pcls.points_packed(), idx, meshes, sign_tensor, nearest, dist, stats)
        signed_dist = dists * sign_tensor
        signed_dist *= 5
        # # dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12) * 2 - 1 
        signed_dist = torch.clamp(signed_dist, -1, 1)
        signed_dist = signed_dist.unsqueeze(-1) # (P,1)
        # uv_query[:, 0] = (uv_query[:, 0] - uv_query[:, 0].min()) / (uv_query[:, 0].max() - uv_query[:, 0].min() + 1e-12) * 2 - 1 
        # uv_query[:, 1] = (uv_query[:, 1] - uv_query[:, 1].min()) / (uv_query[:, 1].max() - uv_query[:, 1].min() + 1e-12) * 2 - 1 
        # dists = torch.clamp(dists, -1, 1)
        uv_query = torch.clamp(uv_query * 2 -1, -1, 1)
        idxs = idxs.reshape(B,-1) # (B,P)
        idxs = idxs - tris_first_idx.unsqueeze(-1)
        
        return signed_dist, idxs, baries, uv_query

    @torch.no_grad()
    def _parse_nearest_projection(self, triangles, points, eps=1e-12):

        # store the location of the closest point
        result = torch.zeros_like(points).to(points.device)
        remain = torch.ones(len(points), dtype=bool).to(points.device)

        # get the three points of each triangle
        # use the same notation as RTCD to avoid confusion
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]

        # check if P is in vertex region outside A
        ab = b - a
        ac = c - a
        ap = points - a
        # this is a faster equivalent of:
        # diagonal_dot(ab, ap)

        d1 = torch.sum(ab * ap, dim=-1)
        d2 = torch.sum(ac * ap, dim=-1)

        # is the point at A
        is_a = torch.logical_and(d1 < eps, d2 < eps)
        if torch.any(is_a):
            result[is_a] = a[is_a]
            remain[is_a] = False

        # check if P in vertex region outside B
        bp = points - b
        d3 = torch.sum(ab * bp, dim=-1)
        d4 = torch.sum(ac * bp, dim=-1)

        # do the logic check
        is_b = (d3 > -eps) & (d4 <= d3) & remain
        if torch.any(is_b):
            result[is_b] = b[is_b]
            remain[is_b] = False

        # check if P in edge region of AB, if so return projection of P onto A
        vc = (d1 * d4) - (d3 * d2)
        is_ab = ((vc < eps) &
                 (d1 > -eps) &
                 (d3 < eps) & remain)
        if torch.any(is_ab):
            v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).view((-1, 1))
            result[is_ab] = a[is_ab] + (v * ab[is_ab])
            remain[is_ab] = False

        # check if P in vertex region outside C
        cp = points - c
        d5 = torch.sum(ab * cp, dim=-1)
        d6 = torch.sum(ac * cp, dim=-1)
        is_c = (d6 > -eps) & (d5 <= d6) & remain
        if torch.any(is_c):
            result[is_c] = c[is_c]
            remain[is_c] = False

        # check if P in edge region of AC, if so return projection of P onto AC
        vb = (d5 * d2) - (d1 * d6)
        is_ac = (vb < eps) & (d2 > -eps) & (d6 < eps) & remain
        if torch.any(is_ac):
            w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).view((-1, 1))
            result[is_ac] = a[is_ac] + w * ac[is_ac]
            remain[is_ac] = False

        # check if P in edge region of BC, if so return projection of P onto BC
        va = (d3 * d6) - (d5 * d4)
        is_bc = ((va < eps) &
                 ((d4 - d3) > - eps) &
                 ((d5 - d6) > -eps) & remain)
        if torch.any(is_bc):
            d43 = d4[is_bc] - d3[is_bc]
            w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).view((-1, 1))
            result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
            remain[is_bc] = False

        # any remaining points must be inside face region
        if torch.any(remain):
            # point is inside face region
            denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
            v = (vb[remain] * denom).reshape((-1, 1))
            w = (vc[remain] * denom).reshape((-1, 1))
            # compute Q through its barycentric coordinates
            result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

        stats = {
            'is_a': is_a,
            'is_b': is_b,
            'is_c': is_c,
            'is_bc': is_bc,
            'is_ac': is_ac,
            'is_ab': is_ab,
            'remain': remain
        }

        return result, stats


    @torch.no_grad()
    def point_mesh_face_distance(self, meshes: Meshes, pcls: Pointclouds):

        if len(meshes) != len(pcls):
            raise ValueError("meshes and pointclouds must be equal sized batches")

        # packed representation for pointclouds
        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_tris = meshes.num_faces_per_mesh().max().item()

        point_to_face, idxs = _C.point_face_dist_forward(
                points, points_first_idx, tris, tris_first_idx, max_points
        )
        
        return point_to_face, idxs


    @torch.no_grad()
    def _calculate_points_inside_meshes_normals(self, points, nearest, triangles, normals_triangles):
        barycentric = self.points_to_barycentric(triangles, nearest)
        normal_at_s = self.barycentric_to_points(normals_triangles, barycentric)
        contains = ((points - nearest) * normal_at_s).sum(1) < 0.0
        contains = 1 - 2*contains
        return contains

    @torch.no_grad()
    def points_to_barycentric(self, triangles, points):

        def diagonal_dot(a, b):
            return torch.matmul(a * b, torch.ones(a.shape[1]).to(a.device))
        
        
        edge_vectors = triangles[:, 1:] - triangles[:, :1]
        w = points - triangles[:, 0].view((-1, 3))

        dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
        dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
        dot02 = diagonal_dot(edge_vectors[:, 0], w)
        dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
        dot12 = diagonal_dot(edge_vectors[:, 1], w)

        inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)

        barycentric = torch.zeros(len(triangles), 3).to(points.device)
        barycentric[:, 2] = (dot00 * dot12 - dot01 *
                            dot02) * inverse_denominator
        barycentric[:, 1] = (dot11 * dot02 - dot01 *
                            dot12) * inverse_denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]

        return barycentric


    @torch.no_grad()
    def barycentric_to_points(self, triangles, barycentric):
        return (triangles * barycentric.view((-1, 3, 1))).sum(dim=1)

    @torch.no_grad()
    def get_barycentric(self, points, closest_faces):
        # points (P,3) point clouds
        # closest_faces (P,3,3) clostest faces
        v1 = closest_faces[:,0] # (P,3)
        v2 = closest_faces[:,1]
        v3 = closest_faces[:,2]

        vq = points - v1

        r31r31 = torch.sum((v3-v1)**2,dim=-1) #(P,)
        r21r21 = torch.sum((v2-v1)**2,dim=-1)
        r21r31 = torch.sum((v2-v1)*(v3-v1),dim=-1)
        r31vq = torch.sum((v3-v1)*vq,dim=-1)
        r21vq = torch.sum((v2-v1)*vq,dim=-1)

        d = r31r31*r21r21 - r21r31**2
        d = torch.clamp(d, 1e-12)
        bary3 = torch.div(r21r21*r31vq - r21r31*r21vq,d)
        bary2 = torch.div(r31r31*r21vq - r21r31*r31vq,d)
        bary1 = 1. - bary2 - bary3

        bary = torch.stack([bary1,bary2,bary3],dim=-1) #(P,3)
        # print(torch.min(bary))
        # print(torch.max(bary))

        return bary


# p = ProjectXYZToUVD()

"""
print(p.faces_to_corres_edges.shape) torch.Size([14372, 3])
print(p.edges_to_corres_faces.shape) torch.Size([21959, 2])
print(p.verts_to_corres_faces.shape) torch.Size([7588, 9])
"""

"""

# ######
# print(p.faces)
# ['__doc__', '__slots__', '_fields', '_field_defaults', '__new__', '_make', '_replace', '__repr__', '_asdict', '__getnewargs__', '__match_args__', 'verts_idx', 'normals_idx', 'textures_idx', 'materials_idx', '__module__', '__hash__', '__getattribute__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__iter__', '__len__', '__getitem__', '__add__', '__mul__', '__rmul__', '__contains__', 'index', 'count', '__class_getitem__', '__str__', '__setattr__', '__delattr__', '__init__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']
# ######

print(p.verts.shape)  torch.Size([7588, 3])

print(p.faces_idx.shape) torch.Size([14372, 3])

print(p.verts_uvs.shape) torch.Size([7588, 2])

print(p.faces_t.shape) torch.Size([14372, 3])

print(torch.all(p.faces_idx == p.faces_t)) tensor(True, device='cuda:0')

"""