import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr

import mcubes
import raymarching
from meshutils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
from .utils import custom_meshgrid, safe_normalize, maxmin_normalize, rodrigues
import pickle

from dmdrive.model.utils import make_aligned, to_homogeneous, get_dmtet_weights, MANOHand, batch_rodrigues, get_dmtet_expression, make_aligned_real_hand
from dmdrive.model.handy import handy_lbs
import trimesh
import smplx
from dmdrive.smplx_model.smplx import smplx_lbs, convert_between


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

def read_gt_smplx(smplx_DIR, device):
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
    global_orient = torch.zeros((1,3))
    full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                            body_pose.reshape(-1, 21, 3),
                            jaw_pose.reshape(-1, 1, 3),
                            leye_pose.reshape(-1, 1, 3),
                            reye_pose.reshape(-1, 1, 3),
                            left_hand_pose.reshape(-1, 15, 3),
                            right_hand_pose.reshape(-1, 15, 3)],
                            dim=1).reshape(-1, 165)
    R_global = rodrigues(global_orient.unsqueeze(0).numpy()).astype(np.float32)

    return betas.to(device), full_pose.to(device), transl.to(device), torch.tensor(R_global).to(device)


class DMTet():
    def __init__(self, device):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [ 1,  0,  2, -1, -1, -1],
            [ 4,  0,  3, -1, -1, -1],
            [ 1,  4,  2,  1,  3,  4],
            [ 3,  1,  5, -1, -1, -1],
            [ 2,  3,  0,  2,  5,  3],
            [ 1,  4,  0,  1,  5,  4],
            [ 4,  2,  5, -1, -1, -1],
            [ 4,  5,  2, -1, -1, -1],
            [ 4,  1,  0,  4,  5,  1],
            [ 3,  2,  0,  3,  5,  2],
            [ 1,  3,  5, -1, -1, -1],
            [ 4,  1,  2,  4,  3,  1],
            [ 3,  0,  4, -1, -1, -1],
            [ 2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=device)
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device=device)
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device=device)
    
    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # pos_nx3: [N, 3]
        # sdf_n:   [N]
        # tet_fx4: [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1) # [F,]
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device=self.device)
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        return verts, faces

def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

@torch.cuda.amp.autocast(enabled=False)
def normal_consistency(face_normals, t_pos_idx):

    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))


def laplacian_uniform(verts, faces):

    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()


@torch.cuda.amp.autocast(enabled=False)
def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.max_level = None
        self.dmtet = opt.dmtet
        self.cuda_ray = opt.cuda_ray
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh
        self.workspace = opt.workspace

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.glctx = None

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
        
        if self.opt.dmtet:
            # load dmtet vertices
            tets = np.load('tets/{}_tets.npz'.format(self.opt.tet_grid_size))
            self.verts = - torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * 2 # covers [-1, 1]
            self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
            self.tet_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
            self.dmtet = DMTet('cuda')

            # vert sdf and deform
            sdf = torch.nn.Parameter(torch.zeros_like(self.verts[..., 0]), requires_grad=True)
            self.register_parameter('sdf', sdf)
            deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
            self.register_parameter('deform', deform)

            edges = torch.tensor([0,1, 0,2, 0,3, 1,2, 1,3, 2,3], dtype=torch.long, device="cuda") # six edges for each tetrahedron.
            all_edges = self.indices[:,edges].reshape(-1,2) # [M * 6, 2]
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

            self.gt_smplx = opt.gt_smplx

            self.tpose = opt.tpose
            self.expression = opt.expression

            if self.opt.h <= 2048 and self.opt.w <= 2048:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = dr.RasterizeGLContext()
           
    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        
        if self.opt.density_activation == 'exp':
            # if self.converter is not None:
            #     g = self.opt.blob_density * torch.exp(- x[:, -1] / (2 * self.opt.blob_radius ** 2))
            # else:
            g = self.opt.blob_density * torch.exp(- d / (2 * self.opt.blob_radius ** 2))
        else:
            g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not (self.cuda_ray):
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):

        if self.opt.dmtet:

            sdf = self.sdf
            deform = torch.tanh(self.deform) / self.opt.tet_grid_size

            vertices, triangles = self.dmtet(self.verts + deform, sdf, self.indices)

            vertices = vertices.detach().cpu().numpy()
            triangles = triangles.detach().cpu().numpy()

        else:

            if resolution is None:
                resolution = self.grid_size

            if self.cuda_ray:
                density_thresh = min(self.mean_density, self.density_thresh) \
                    if np.greater(self.mean_density, 0) else self.density_thresh
            else:
                density_thresh = self.density_thresh
            
            # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
            if self.opt.density_activation == 'softplus':
                density_thresh = density_thresh * 25
            
            sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

            # query
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                        val = self.density(pts.to(self.aabb_train.device))
                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

            print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
            vertices = vertices / (resolution - 1.0) * 2 - 1

        # clean
        vertices = vertices.astype(np.float32)
        vertices[:, 0] *= -1
        triangles = triangles.astype(np.int32)
        triangles = triangles[:, [0,2,1]]
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)
        
        # decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(os.path.join(path, f'mesh.ply'))

        def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            # v, f: torch Tensor
            device = v.device
            v_np = v.cpu().numpy() # [N, 3]
            f_np = f.cpu().numpy() # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uvs
            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4 # for faster unwrap...
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0
            
            if self.glctx is None:
                if h <= 2048 and w <= 2048:
                    self.glctx = dr.RasterizeCudaContext()
                else:
                    self.glctx = dr.RasterizeGLContext()

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask] # [M, 3]

                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_feats.append(results_['albedo'].float())
                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)
            
            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as an antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

            # do ssaa after the NN search, in numpy
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f'{name}mesh.obj')
            mtl_file = os.path.join(path, f'{name}mesh.mtl')

            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:
                fp.write(f'mtllib {name}mesh.mtl \n')
                
                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
            
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n') 

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl mat0 \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}albedo.png \n')

        _export(v, f)

    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        # nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # nears.unsqueeze_(-1)
        # fars.unsqueeze_(-1)
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device)) # [N, 3]

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, self.opt.num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / self.opt.num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        if self.converter is not None:
            dists, idxs, baries, uv_query = self.converter.point_to_face_coord_naive(xyzs.reshape(-1, 3))
            uvd = torch.cat((uv_query, dists), -1)
            uvd = uvd.reshape(xyzs.shape)
            density_outputs = self.density(uvd.reshape(-1, 3))
        else:
            density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, self.opt.num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, self.opt.num_steps, -1)

        # upsample z_vals (nerf-like)
        if self.opt.upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], self.opt.upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

                if self.converter is not None:
                    dists, idxs, baries, uv_query = self.converter.point_to_face_coord_naive(new_xyzs.reshape(-1, 3))
                    new_uvd = torch.cat((uv_query, dists), -1)
                    new_uvd = new_uvd.reshape(new_xyzs.shape)
            
            # only forward new points to save computation
            if self.converter is not None:
                new_density_outputs = self.density(new_uvd.reshape(-1, 3))
            else:
                new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, self.opt.upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, self.opt.upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))
            if self.converter is not None:
                uvd = torch.cat([uvd, new_uvd], dim=1)
                uvd = torch.gather(uvd, dim=1, index=z_index.unsqueeze(-1).expand_as(uvd))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        light_d = light_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        dirs = safe_normalize(dirs)
        if self.converter is not None:
            shading = 'albedo'
            sigmas, rgbs, normals = self(uvd.reshape(-1, 3), dirs.reshape(-1, 3), light_d.reshape(-1, 3), ratio=ambient_ratio, shading=shading)
        else:
            sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d.reshape(-1, 3), ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]
        if normals is not None:
            normals = normals.view(N, -1, 3)

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        weights_sum = weights_sum.reshape(*prefix)

        if self.training:
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.sum(-1).mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if (self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0) and normals is not None:
                normal_image = torch.sum(weights.unsqueeze(-1) * (normals + 1) / 2, dim=-2) # [N, 3], in [0, 1]
                results['normal_image'] = normal_image
        
        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum

        return results


    def run_cuda(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device)) # [N, 3]

        results = {}

        if self.converter is not None:
            shading = 'albedo'
        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, self.opt.dt_gamma, self.opt.max_steps)
            dirs = safe_normalize(dirs)

            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                light_d = light_d[flatten_rays]
            
            # scale_factor = 5
            # xyzs *= scale_factor
            if self.converter is not None:
                dists, idxs, baries, uv_query = self.converter.point_to_face_coord_naive(xyzs)
                uvd = torch.cat((uv_query, dists), -1)
                sigmas, rgbs, normals = self(uvd, dirs, light_d, ratio=ambient_ratio, shading=shading)
                # ## save xyz and uvd
                # flag = np.random.randint(0,99)
                # xyz_to_save = xyzs.reshape(-1,3).clone().cpu().numpy()
                # uvd_to_save = uvd.reshape(-1,3).clone().cpu().numpy()
                # np.save('check_uvd/xyz_{}.npy'.format(flag), xyz_to_save)
                # np.save('check_uvd/uvd_{}.npy'.format(flag), uvd_to_save)            
            else:
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, binarize)
            
            # normals related regularizations
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss 
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if (self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0) and normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), (normals + 1) / 2, ts, rays, T_thresh, binarize)
                results['normal_image'] = normal_image
            
            # weights normalization
            results['weights'] = weights

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            # weights_sum_n = weights_sum.clone()
            # depth_n = depth.clone()
            # normal_image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < self.opt.max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, self.opt.dt_gamma, self.opt.max_steps)
                dirs = safe_normalize(dirs)
                # xyzs = torch.tanh(xyzs)
                # xyzs = torch.pow(xyzs, s3)
                if self.converter is not None:
                    dists, idxs, baries, uv_query = self.converter.point_to_face_coord_naive(xyzs)
                    uvd = torch.cat((uv_query, dists), -1)
                    sigmas, rgbs, normals = self(uvd, dirs, light_d, ratio=ambient_ratio, shading=shading)
                else:
                    sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                
                # if normals is not None:
                #     raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, (normals + 1) / 2, ts, weights_sum_n, depth_n, normal_image, T_thresh, binarize)
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, binarize)

                rays_alive = rays_alive[rays_alive >= 0]
                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        depth = depth.view(*prefix)
        depth = maxmin_normalize(depth)

        weights_sum = weights_sum.reshape(*prefix)
        
        # normal_image = normal_image.view(*prefix, 3)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum

        # results['normal_image'] = normal_image
        
        return results

    @torch.no_grad()
    def init_tet(self, mesh=None):

        if mesh is not None:
            # normalize mesh
            # scale = 0.8 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
            # center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
            # mesh.vertices = (mesh.vertices - center) * scale

            # init scale
            # self.tet_scale = torch.from_numpy(np.abs(mesh.vertices).max(axis=0) + 1e-1).to(self.verts.dtype).cuda()
            self.tet_scale = torch.from_numpy(np.array([np.abs(mesh.vertices).max(axis=0)]) + 1e-1).to(self.verts.dtype).cuda()
            self.verts = self.verts * self.tet_scale

            # init sdf
            import cubvh
            BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
            sdf, _, _ = BVH.signed_distance(self.verts, return_uvw=False, mode='watertight')
            sdf *= -10 # INNER is POSITIVE, also make it stronger
            self.sdf.data += sdf.to(self.sdf.data.dtype).clamp(-1, 1)

        else:

            if self.cuda_ray:
                density_thresh = min(self.mean_density, self.density_thresh)
                # stage1: mean:19999999, thresh:10
                # stage2: mean:0.9987, thresh:10
            else:
                density_thresh = self.density_thresh
        
            if self.opt.density_activation == 'softplus':
                density_thresh = density_thresh * 25

            # init scale
            sigma = self.density(self.verts)['sigma'] # verts covers [-1, 1] now
            mask = sigma > density_thresh
            valid_verts = self.verts[mask]
            self.tet_scale = valid_verts.abs().amax(dim=0) + 1e-1
            self.verts = self.verts * self.tet_scale

            # init sigma
            sigma = self.density(self.verts)['sigma'] # new verts
            self.sdf.data += (sigma - density_thresh).clamp(-1, 1)

        print(f'[INFO] init dmtet: scale = {self.tet_scale}')


    def run_dmtet(self, rays_o, rays_d, mvp, h, w, poses=None, bbox=None, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, dst_pose=None, dist_expr=None, **kwargs):
        # mvp: [B, 4, 4]

        device = mvp.device
        campos = rays_o[:, 0, :] # only need one ray per batch

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(campos + torch.randn_like(campos)).view(-1, 1, 1, 3) # [B, 1, 1, 3]

        results = {}

        # get mesh
        sdf = self.sdf
        deform = torch.tanh(self.deform) / self.opt.tet_grid_size

        verts, faces = self.dmtet(self.verts + deform, sdf, self.indices)

        if dst_pose is not None:
            if self.handy_model is not None or self.mano_model is not None:
                camera_offset = torch.tensor([0,0.07,0]).reshape(1,3).to(verts.device)
                
                # make dmtet mesh aligned to Handy format     
                verts[:,0] *= -1
                faces = faces[:, [0,2,1]]
                dst_shape = None
                dst_pose = dst_pose.reshape(1,-1)
                if dst_pose.shape[-1] == 58:
                    dst_shape = dst_pose[...,:10].reshape(1,10)
                    dst_pose = dst_pose[...,10:].reshape(1,16,3)
                elif dst_pose.shape[-1] == 48:
                    dst_pose = torch.tensor(dst_pose, dtype=torch.float32, device=device).reshape(1,16,3)
                else:
                    raise ValueError("Expected shape is (x, 48) or (x, 58), but got" + str(dst_pose.shape))

                ## TODO: modify hard-coded canonical shape
                if dst_shape is None:
                    if self.opt.hand_dst_shape is not None:
                        file_extension = self.opt.hand_dst_shape.split('.')[-1].lower()
                        if file_extension == 'npy':
                            dst_shape = np.load(self.opt.hand_dst_shape)
                            dst_shape = torch.from_numpy(dst_shape[:10]).reshape(1,10).to(torch.float32).to(device)
                        elif file_extension == 'pth':
                            dst_shape = torch.load(self.opt.hand_dst_shape)[0].reshape(1,10).to(torch.float32).to(device)
                        else:
                            raise ValueError("Unsupported file format")
                    else:
                        dst_shape = torch.zeros((1,10), device=device)
                        print(f'[WARNING] Using zero shape model as template!!!')
        
            if self.handy_model is not None:
                v_rest, T_rest2dst, _ = self.handy_model.verts_transformations(
                    return_tensor=True,
                    poses=dst_pose,
                    betas=dst_shape,
                    concat_joints=True
                ) # [B, N, 3], [B, N, 4, 4]
                
                faces_rest = self.handy_model.faces # [f, 3]
                v_rest = v_rest[0]
                joints_rest = v_rest[7231:, :]
                mat_aligned = make_aligned(joints_rest, self.mesh_scale) # including minus joint4
                v_rest = torch.einsum('Ni, Bi->BN', mat_aligned, to_homogeneous(v_rest))[:,:3]
                joints_rest_aligned = v_rest[7231:, :]
                
                ## relocate dmtet mesh's origin to joint4
                verts -= (joints_rest_aligned[4] + camera_offset)
                
                lbs_weights_rest = self.handy_model.lbs_weights
                dm_lbs_weights = get_dmtet_weights(verts, v_rest, faces_rest, lbs_weights_rest)
                
                verts_to_deform = torch.einsum('Ni, Bi->BN', torch.inverse(mat_aligned), to_homogeneous(verts))[:,:3]
                
                
                ##TODO:interpolate posedirs and shapedirs
                
                T_handy_rest2dst, verts_deform, joints_deform = handy_lbs(betas=dst_shape, pose=dst_pose, v_template=verts_to_deform, shapedirs=self.handy_model.shapedirs,
                                posedirs=self.handy_model.posedirs, joints=joints_rest.unsqueeze(0), 
                                parents=self.handy_model.parents, lbs_weights=dm_lbs_weights)
                verts_deform = verts_deform[0]
                joints_deform = joints_deform[0]
                mat_aligned_dm = make_aligned(joints_deform, self.mesh_scale)
                verts_deform = torch.einsum('Ni, Bi->BN', mat_aligned_dm, to_homogeneous(verts_deform))[:,:3].float()
                ## translate all meshes back to original space
                verts += (joints_rest_aligned[4] + camera_offset)
                verts_deform += camera_offset

            elif self.mano_model is not None:
                if self.opt.hand_tpose:
                    zero_pose = torch.zeros_like(dst_pose)
                    zero_shape = torch.zeros_like(dst_shape)
                    theta_rodrigues = batch_rodrigues(zero_pose.reshape(-1, 3)).reshape(1, 16, 3, 3)
                    __theta = theta_rodrigues.reshape(1, 16, 3, 3)
                    so = self.mano_model(betas = dst_shape.reshape(1, 10), hand_pose = __theta[:, 1:], global_orient = __theta[:, 0].view(1, 1, 3, 3))
                    v_rest = so['vertices'].clone().reshape(1, -1, 3) / self.real_shape_scale
                    joints = so['joints'].clone().reshape(-1,3) / self.real_shape_scale
                    # root = deepcopy(joints[4])
                    faces_rest = self.mano_model.faces # [f, 3]
                    faces_rest = faces_rest.astype(np.int32)
                    v_rest = v_rest[0]
                    joints_rest = joints
                    mat_aligned = make_aligned(joints_rest, self.mesh_scale) # including minus joint4
                    v_rest = torch.einsum('Ni, Bi->BN', mat_aligned, to_homogeneous(v_rest))[:,:3]
                    joints_rest_aligned = torch.einsum('Ni, Bi->BN', mat_aligned, to_homogeneous(joints_rest))[:,:3]
                    
                    
                    ## relocate dmtet mesh's origin to joint4
                    verts -= (joints_rest_aligned[4] + camera_offset)
                    
                    lbs_weights_rest = self.mano_model.lbs_weights
                    dm_lbs_weights = get_dmtet_weights(verts, v_rest, faces_rest, lbs_weights_rest)
                    # trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy(), process=False).export("/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/check_mesh/dmtet_verts.obj")
                    # trimesh.Trimesh(v_rest.detach().cpu().numpy(), faces_rest, process=False).export("/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/check_mesh/rest_verts_dstshape.obj")
                    # trimesh.Trimesh(v_rest.detach().cpu().numpy(), faces_rest, process=False).export("/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/check_mesh/rest_verts_zeroshape.obj")
                    verts_to_deform = torch.einsum('Ni, Bi->BN', torch.inverse(mat_aligned), to_homogeneous(verts))[:,:3]
                    
                    
                    ##TODO:interpolate posedirs and shapedirs
                    
                    T_handy_rest2dst, verts_deform, joints_deform = handy_lbs(betas=torch.zeros((1,10),device=device,dtype=torch.float32), pose=dst_pose, v_template=verts_to_deform, shapedirs=self.mano_model.shapedirs,
                                    posedirs=self.mano_model.posedirs, joints=joints_rest.unsqueeze(0), 
                                    parents=self.mano_model.parents, lbs_weights=dm_lbs_weights)
                    verts_deform = verts_deform[0]
                    joints_deform = joints_deform[0]
                    if self.opt.real_hand_pose:
                        mat_aligned_dm = make_aligned_real_hand(joints_deform, self.mesh_scale)
                    else:
                        mat_aligned_dm = make_aligned(joints_deform, self.mesh_scale)
                    verts_deform = torch.einsum('Ni, Bi->BN', mat_aligned_dm, to_homogeneous(verts_deform))[:,:3].float()
                    ## translate all meshes back to original space
                    verts += (joints_rest_aligned[4] + camera_offset)
                    verts_deform += camera_offset
                    # trimesh.Trimesh(verts_deform.detach().cpu().numpy(), faces.detach().cpu().numpy(), process=False).export("check_mesh/verts_deform1.obj")
                else:
                    zero_pose = torch.zeros_like(dst_pose)
                    theta_rodrigues = batch_rodrigues(zero_pose.reshape(-1, 3)).reshape(1, 16, 3, 3)
                    __theta = theta_rodrigues.reshape(1, 16, 3, 3)
                    so = self.mano_model(betas = dst_shape.reshape(1, 10), hand_pose = __theta[:, 1:], global_orient = __theta[:, 0].view(1, 1, 3, 3))
                    v_rest = so['vertices'].clone().reshape(1, -1, 3) / self.real_shape_scale
                    joints = so['joints'].clone().reshape(-1,3) / self.real_shape_scale
                    # root = deepcopy(joints[4])
                    faces_rest = self.mano_model.faces # [f, 3]
                    faces_rest = faces_rest.astype(np.int32)
                    v_rest = v_rest[0]
                    joints_rest = joints
                    mat_aligned = make_aligned(joints_rest, self.mesh_scale) # including minus joint4
                    v_rest = torch.einsum('Ni, Bi->BN', mat_aligned, to_homogeneous(v_rest))[:,:3]
                    joints_rest_aligned = torch.einsum('Ni, Bi->BN', mat_aligned, to_homogeneous(joints_rest))[:,:3]
                    
                    ## relocate dmtet mesh's origin to joint4
                    verts -= (joints_rest_aligned[4] + camera_offset)
                    
                    lbs_weights_rest = self.mano_model.lbs_weights
                    dm_lbs_weights = get_dmtet_weights(verts, v_rest, faces_rest, lbs_weights_rest)
                    
                    verts_to_deform = torch.einsum('Ni, Bi->BN', torch.inverse(mat_aligned), to_homogeneous(verts))[:,:3]
                    
                    
                    ##TODO:interpolate posedirs and shapedirs
                    
                    T_handy_rest2dst, verts_deform, joints_deform = handy_lbs(betas=dst_shape, pose=dst_pose, v_template=verts_to_deform, shapedirs=self.mano_model.shapedirs,
                                    posedirs=self.mano_model.posedirs, joints=joints_rest.unsqueeze(0), 
                                    parents=self.mano_model.parents, lbs_weights=dm_lbs_weights)
                    verts_deform = verts_deform[0]
                    joints_deform = joints_deform[0]
                    mat_aligned_dm = make_aligned(joints_deform, self.mesh_scale)
                    verts_deform = torch.einsum('Ni, Bi->BN', mat_aligned_dm, to_homogeneous(verts_deform))[:,:3].float()
                    ## translate all meshes back to original space
                    verts += (joints_rest_aligned[4] + camera_offset)
                    verts_deform += camera_offset

            elif self.smplx_model is not None:

                def save_mesh(vertices, faces, path):
                    """
                    save mesh to path in .ply format
                    """

                    mesh = trimesh.Trimesh(vertices, faces)
                    mesh.export(os.path.join(path))
                    print(f'saved mesh to {path}')
                
                # camera offset using pyrender
                camera_offset = torch.tensor([0,0,0]).reshape(1,3).to(verts.device)
                
                # make dmtet mesh aligned to Handy format     
                verts[:,0] *= -1
                faces = faces[:, [0,2,1]]

                # save_mesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy(),'verts.ply')
                
                ## TODO: modify hard-coded canonical shape
                zero_shape = torch.zeros((1,10), device=device)
                if torch.is_tensor(dst_pose):
                    dst_pose = torch.tensor(dst_pose.detach().cpu().numpy(), dtype=torch.float32, device=device).reshape(1,-1)
                else:
                    dst_pose = torch.tensor(dst_pose, dtype=torch.float32, device=device).reshape(1,-1)
                # dst_pose = torch.tensor(dst_pose, dtype=torch.float32, device=device).reshape(1,165)
                # dst_pose[:,:3] = torch.zeros((1,3), device=device)                
                betas, gt_pose, transl, R_global = read_gt_smplx(self.gt_smplx, device=device)# [B, N, 3], [B, N, 4, 4]
                if self.tpose:
                    gt_pose[:] = 0
                    transl[:] = 0
                # betas [B,10] gt_pose[B,165] transl[B,3] R_global[B,3,3]
                v_rest, joints_rest = self.smplx_model(
                    gt_pose, betas, transl=transl,
                    return_tensor=True,
                    return_joints=True
                )# [bs,n,3] no expression 
                # transorform dmtet mesh to smplx coord; if global_pose is (0,0,0), R_global is ones(3) which can be omitted
                # v_rest = torch.bmm(v_rest , R_global).type_as(verts) # [B,N,3]
                # joints_rest = torch.bmm(joints_rest , R_global).type_as(verts) 
                # gt_pose[:,:3] = torch.zeros((1,3), device=device)

                # print('v_rest', v_rest.shape, gt_pose.shape, verts.shape)

                scale = 0.5
                root = joints_rest[0,0,:].reshape(1,3)
                verts_to_deform = verts/scale + root #[N,3]

                ## verified
 
                faces_rest = self.smplx_model.faces.astype(np.int32) # [f, 3]
                v_rest = v_rest[0]

                lbs_weights_rest = self.smplx_model.lbs_weights
                dm_lbs_weights = get_dmtet_weights(verts_to_deform, v_rest, faces_rest, lbs_weights_rest)
                if self.expression:
                    expr_dirs = self.smplx_model.expr_dirs
                    dm_expr_dirs = get_dmtet_expression(verts, v_rest, faces_rest, expr_dirs)
                    dst_expr = dst_pose[..., 165:]
                    dst_pose = dst_pose[..., :165]
                    # print('dst_pose', dst_pose)
                    # print('dst_expr', dst_expr)
                else:
                    dst_expr = None
                    dm_expr_dirs = None
                    dst_pose = dst_pose[..., :165]
                
                # save_mesh(verts_to_deform.detach().cpu().numpy(), faces.detach().cpu().numpy(),'verts_to_deform.ply')
                if self.tpose: #if gt_pose is tpose, no need to deform back first. 
                    T_handy_rest2dst, verts_deform, joints_deform = smplx_lbs(betas=zero_shape, pose=dst_pose.unsqueeze(0), 
                                    v_template=verts_to_deform.unsqueeze(0), shapedirs=self.smplx_model.shapedirs, expr_dirs = dm_expr_dirs,
                                    posedirs=self.smplx_model.posedirs, joints=joints_rest, expression = dst_expr,
                                    parents=self.smplx_model.parents, lbs_weights=dm_lbs_weights, pose2rot=True)
                    
                    v_rest, joints_rest = self.smplx_model(
                        dst_pose, betas, transl=transl,
                        return_tensor=True,
                        return_joints=True
                    )
                    root_dst = joints_rest[0,0,:].reshape(1,3)
                # dst_pose[:, 75:] = self.gt_pose[:, 75:]
                # print(self.gt_pose[:,:75])
                # else:
                #     convert_mtx = convert_between(gt_pose, dst_pose) #
                    
                #     T_handy_rest2dst, verts_deform, joints_deform = smplx_lbs(betas=zero_shape, pose=convert_mtx.unsqueeze(0), 
                #                     v_template=verts_to_deform.unsqueeze(0), shapedirs=self.smplx_model.shapedirs,
                #                     posedirs=self.smplx_model.posedirs, joints=joints_rest, 
                #                     parents=self.smplx_model.parents, lbs_weights=dm_lbs_weights, pose2rot=False)
                    
                #     T_handy_rest2dst, verts_deform, joints_deform = smplx_lbs(betas=zero_shape, pose=dst_pose.unsqueeze(0), 
                #                     v_template=verts_deform, shapedirs=self.smplx_model.shapedirs,
                #                     posedirs=self.smplx_model.posedirs, joints=joints_deform, 
                #                     parents=self.smplx_model.parents, lbs_weights=dm_lbs_weights, pose2rot=True)
                
                
                verts_deform = verts_deform[0]
                joints_deform = joints_deform[0]

                ## translate all meshes back to original space
                # verts_deform = verts_deform + transl2 #[B,N,3]
                verts_deform = verts_deform - root_dst #[B,N,3]
                verts_deform = verts_deform*scale
                
                # # check mesh
                # if self.opt.debug:
                #     save_path = os.path.join(self.workspace, 'mesh_check')
                #     os.makedirs(save_path, exist_ok=True)
                #     save_mesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy().astype(np.int32), os.path.join(save_path, 'dmtet.ply'))
                #     save_mesh(verts_deform.detach().cpu().numpy(), faces.detach().cpu().numpy().astype(np.int32),os.path.join(save_path, 'dmtet_deform.ply'))

            else:
                raise Exception("Need a template model to drive the mesh")

            # flip again to sample feature
            verts[:,0] *= -1
            faces = faces[:, [0,2,1]]
            verts_deform[:,0] *= -1
     
        if dst_pose is not None:
            if self.opt.hand_tpose:
                __i0, __i1, __i2 = faces[:, 0], faces[:, 1], faces[:, 2]
                __v0, __v1, __v2 = verts[__i0, :], verts[__i1, :], verts[__i2, :]
                
                __face_normals = torch.cross(__v1 - __v0, __v2 - __v0)
                __face_normals = safe_normalize(__face_normals)
                
                __vn = torch.zeros_like(verts)
                __vn.scatter_add_(0, __i0[:, None].repeat(1,3), __face_normals)
                __vn.scatter_add_(0, __i1[:, None].repeat(1,3), __face_normals)
                __vn.scatter_add_(0, __i2[:, None].repeat(1,3), __face_normals)

                __vn = torch.where(torch.sum(__vn * __vn, -1, keepdim=True) > 1e-20, __vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=__vn.device))
            i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
            v0, v1, v2 = verts_deform[i0, :], verts_deform[i1, :], verts_deform[i2, :]

            faces = faces.int()
            
            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = safe_normalize(face_normals)
            
            vn = torch.zeros_like(verts_deform)
            vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

            vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

            # rasterization
            verts_clip = torch.bmm(F.pad(verts_deform, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1), mvp.permute(0,2,1)).float()  # [B, N, 4]
            rast, rast_db = dr.rasterize(self.glctx, verts_clip, faces, (h, w))
            verts_camcoord = torch.bmm(verts_deform.unsqueeze(0), torch.inverse(poses[:, :3, :3])).float() 
            alpha = (rast[..., 3:] > 0).float()
            xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces) # [B, H, W, 3]
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
            normal = safe_normalize(normal)

            xyzs = xyzs.view(-1, 3)
            mask = (rast[..., 3:] > 0).view(-1).detach()

            # do the lighting here since we have normal from mesh now.
            albedo = torch.zeros_like(xyzs, dtype=torch.float32)
            if mask.any():
                masked_albedo = self.density(xyzs[mask])['albedo']
                albedo[mask] = masked_albedo.float()
            albedo = albedo.view(-1, h, w, 3)

            # these two modes lead to no parameters to optimize if using --lock_geo.
            if self.opt.lock_geo and shading in ['textureless', 'normal']:
                shading = 'lambertian'

            if shading == 'albedo':
                color = albedo
            elif shading == 'textureless':
                lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
                color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
                color = albedo * lambertian.unsqueeze(-1)

            color = dr.antialias(color, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
            alpha = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 1]
            # depth = rast[:, :, :, [2]] # [B, H, W]
            verts_depth, _ = dr.interpolate(verts_camcoord, rast, faces)
            depth = -1 * verts_depth[:,:,:,[2]] # [B, H, W]
            mask = (rast[..., 3:] > 0).detach()
            depth = maxmin_normalize(depth)
            depth[~mask] = 0

        else:
        # get normals
            i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
            v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

            faces = faces.int()
            
            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = safe_normalize(face_normals)
            
            vn = torch.zeros_like(verts)
            vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

            vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

            # rasterization
            verts_clip = torch.bmm(F.pad(verts, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1), 
                                mvp.permute(0,2,1)).float()  # [B, N, 4]
            rast, rast_db = dr.rasterize(self.glctx, verts_clip, faces, (h, w))
            verts_camcoord = torch.bmm(verts.unsqueeze(0), torch.inverse(poses[:, :3, :3])).float() 
            alpha = (rast[..., 3:] > 0).float()
            xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces) # [B, H, W, 3]
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
            normal = safe_normalize(normal)

            xyzs = xyzs.view(-1, 3)
            mask = (rast[..., 3:] > 0).view(-1).detach()
            
            # do the lighting here since we have normal from mesh now.
            albedo = torch.zeros_like(xyzs, dtype=torch.float32)
            if mask.any():
                masked_albedo = self.density(xyzs[mask])['albedo']
                albedo[mask] = masked_albedo.float()
            albedo = albedo.view(-1, h, w, 3)

            # these two modes lead to no parameters to optimize if using --lock_geo.
            if self.opt.lock_geo and shading in ['textureless', 'normal']:
                shading = 'lambertian'

            if shading == 'albedo':
                color = albedo
            elif shading == 'textureless':
                lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
                color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
                color = albedo * lambertian.unsqueeze(-1)

            color = dr.antialias(color, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
            alpha = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 1]
            # depth = rast[:, :, :, [2]] # [B, H, W]
            verts_depth, _ = dr.interpolate(verts_camcoord, rast, faces)
            depth = -1 * verts_depth[:,:,:,[2]] # [B, H, W]
            mask = (rast[..., 3:] > 0).detach()
            depth = maxmin_normalize(depth)
            depth[~mask] = 0

        if bbox is not None:

            if kwargs.get('head_recon', False):
                # bbox = [w//2-w//8, 0]
                bbox_size = w//4
            elif kwargs.get('hand_recon', False):
                # bbox = [w//2-w//16, 0]
                bbox_size = w//8
            color = color[:, bbox[1]:bbox[1]+bbox_size, bbox[0]:bbox[0]+bbox_size]
            alpha = alpha[:, bbox[1]:bbox[1]+bbox_size, bbox[0]:bbox[0]+bbox_size]
            depth = depth[:, bbox[1]:bbox[1]+bbox_size, bbox[0]:bbox[0]+bbox_size]
            depth = maxmin_normalize(depth)

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1
        
        if torch.is_tensor(bg_color) and len(bg_color.shape) > 1:
            if kwargs.get('head_recon', False):
                bg_color = bg_color.view(-1, h//4, w//4, 3)
            elif kwargs.get('hand_recon', False):
                bg_color = bg_color.view(-1, h//8, w//8, 3)
            else:
                bg_color = bg_color.view(-1, h, w, 3)
        
        color = color + (1 - alpha) * bg_color
        # cv2.imwrite('tmp/test2.png', (color[0].detach().cpu().numpy()*255).astype('uint8'))

        results['depth'] = depth        
        results['image'] = color
        results['weights_sum'] = alpha.squeeze(-1)

        if self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0 or not self.training:
            normal_image = dr.antialias((normal + 1) / 2, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
            if bbox is not None:
                if kwargs.get('head_recon', False):
                    # bbox = [w//2-w//8, 0]
                    bbox_size = w//4
                elif kwargs.get('hand_recon', False):
                    # bbox = [w//2-w//16, 0]
                    bbox_size = w//8
                normal_image = normal_image[:, bbox[1]:bbox[1]+bbox_size, bbox[0]:bbox[0]+bbox_size]
            results['normal_image'] = normal_image
        
        # regularizations
        if self.training:
            if self.opt.lambda_mesh_normal > 0:
                results['normal_loss'] = normal_consistency(face_normals, faces)
            if self.opt.lambda_mesh_laplacian > 0:
                if self.opt.lap_vn:
                    if self.opt.hand_tpose:
                        results['lap_loss'] = laplacian_smooth_loss(safe_normalize(__vn), faces)
                    else:
                        results['lap_loss'] = laplacian_smooth_loss(safe_normalize(vn), faces)
                else:
                    results['lap_loss'] = laplacian_smooth_loss(verts, faces)

        return results


    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not (self.cuda_ray):
            return 
        
        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        if self.converter is not None:
                            dists, idxs, baries, uv_query = self.converter.point_to_face_coord_naive(cas_xyzs)
                            uvd = torch.cat((uv_query, dists), -1)
                            uvd = uvd.reshape(cas_xyzs.shape)
                            sigmas = self.density(uvd)['sigma'].reshape(-1).detach()
                        else:
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign 
                        tmp_grid[cas, indices] = sigmas
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        if self.cuda_ray:
            self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)
        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')


    def render(self, rays_o, rays_d, mvp, h, w, poses=None, bbox=None, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: pred_rgb: [B, N, 3]
        B, N = rays_o.shape[:2]
        device = rays_o.device

        if self.dmtet:
            results = self.run_dmtet(rays_o, rays_d, mvp, h, w, poses=poses, bbox=bbox, **kwargs)
        elif self.cuda_ray:
            results = self.run_cuda(rays_o, rays_d, **kwargs)
        else:
            if staged:
                depth = torch.empty((B, N), device=device)
                image = torch.empty((B, N, 3), device=device)
                weights_sum = torch.empty((B, N), device=device)

                for b in range(B):
                    head = 0
                    while head < N:
                        tail = min(head + max_ray_batch, N)
                        results_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                        depth[b:b+1, head:tail] = results_['depth']
                        weights_sum[b:b+1, head:tail] = results_['weights_sum']
                        image[b:b+1, head:tail] = results_['image']
                        head += max_ray_batch
                
                results = {}
                results['depth'] = depth
                results['image'] = image
                results['weights_sum'] = weights_sum

            else:
                results = self.run(rays_o, rays_d, **kwargs)

        return results
