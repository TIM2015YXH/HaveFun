import torch
import numpy as np
import torch.nn as nn


class POF_Generator(object):
    def __init__(self, pof_size=[368, 368], pof_conns=np.array([[0, 1], [1, 2]]), dist_thresh=24.0, normalize=True, exploit_uvz=True, k=100, *args, **kargs):
        dist_thresh = max(pof_size[1], pof_size[0]) / 368 * dist_thresh
        self.num_pof, self.pof_size, self.pof_conns, self.dist_thresh=len(pof_conns),pof_size, pof_conns, dist_thresh
        self.x_m       = np.tile(np.arange(pof_size[0]), (self.num_pof, pof_size[1], 1))
        self.y_m       = np.tile(np.arange(pof_size[1]), (self.num_pof, pof_size[0], 1)).transpose([0, 2, 1])
        self.scale     = np.array(pof_size).reshape(1, 2)
        self.normalize = normalize
        self.k = k
        self.exploit_uvz = exploit_uvz

    def __call__(self, key_points_2d, key_points_3d, *args, **kargs):
        pt_xy, pt_valid_2d, pt_xyz, pt_valid_3d = key_points_2d[:, :2]*self.scale, key_points_2d[:, 2], key_points_3d[:, :3], key_points_3d[:, 3]<1
        num_pof, pof_size, pof_conns = self.num_pof, self.pof_size, self.pof_conns
        pt_a_2d, pt_b_2d = pt_xy[pof_conns[:, 0]], pt_xy[pof_conns[:, 1]]
        vab = pt_b_2d - pt_a_2d
        vab = vab / (np.sqrt(np.square(vab).sum(-1)).reshape(-1, 1) + 1e-6)
        dx, dy = self.x_m - pt_a_2d[:, 0].reshape(-1, 1, 1), self.y_m-pt_a_2d[:, 1].reshape(-1, 1, 1)
        dist = np.abs(dy*vab[:, 0].reshape(-1, 1, 1)-dx*vab[:, 1].reshape(-1, 1, 1))

        x_min, x_max = np.minimum(pt_a_2d[:, 0], pt_b_2d[:, 0]) - self.dist_thresh, np.maximum(pt_a_2d[:, 0], pt_b_2d[:, 0]) + self.dist_thresh
        y_min, y_max = np.minimum(pt_a_2d[:, 1], pt_b_2d[:, 1]) - self.dist_thresh, np.maximum(pt_a_2d[:, 1], pt_b_2d[:, 1]) + self.dist_thresh
        x_min, x_max, y_min, y_max = x_min.reshape(-1, 1, 1), x_max.reshape(-1, 1, 1), y_min.reshape(-1, 1, 1), y_max.reshape(-1, 1, 1)

        if not self.exploit_uvz:
            mask = ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[pof_conns[:,0]]<3) * (pt_valid_2d[pof_conns[:,1]]<3) * pt_valid_3d[pof_conns[:,0]] * pt_valid_3d[pof_conns[:,0]]).reshape(-1, 1, 1)).astype(np.float)
        else:
            mask =  ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[pof_conns[:,0]]<3) * (pt_valid_2d[pof_conns[:,1]]<3)).reshape(-1, 1, 1)).astype(np.float)

        pt_a_3d, pt_b_3d = pt_xyz[pof_conns[:, 0]], pt_xyz[pof_conns[:, 1]]
        vab_3d = pt_b_3d - pt_a_3d
        if self.normalize:
            vab_3d = vab_3d / (np.sqrt(np.square(vab_3d).sum(-1)).reshape(-1, 1) +1e-6)

        if not self.exploit_uvz:
            pof_value = vab_3d
        else:
            pof_value = np.concatenate([vab, vab_3d[:,2:3]], 1)

        pof = np.repeat(mask[:, :, :, np.newaxis], 3, axis=3) * pof_value.reshape(-1, 1, 1, 3)
        pof = pof.transpose(0, 3, 1, 2).reshape(-1, pof_size[1], pof_size[0])

        pt_valid_2d = ((pt_valid_2d[pof_conns[:,0]]<3) * (pt_valid_2d[pof_conns[:,1]]<3)).reshape(-1, 1, 1).astype(np.float)
        pt_valid_3d = (pt_valid_3d[pof_conns[:, 0]] * pt_valid_3d[pof_conns[:, 1]]).reshape(-1, 1, 1).astype(np.float)

        if not self.exploit_uvz:
            pof_valid = mask*pt_valid_3d*pt_valid_2d + pt_valid_3d*pt_valid_2d
            pof_valid = pof_valid[..., np.newaxis].repeat(3, axis=3).transpose(0, 3, 1, 2).reshape(-1, pof_size[1], pof_size[0]).astype(np.float)
        else:
            pof_valid_a = np.tile(pt_valid_2d.reshape(-1, 1, 1, 1), (1, pof_size[1], pof_size[0], 2))
            pof_valid_b = np.tile(pt_valid_3d.reshape(-1, 1, 1, 1), (1, pof_size[1], pof_size[0], 1))
            pof_valid = np.concatenate([pof_valid_a, pof_valid_b], 3).transpose(0, 3, 1, 2).reshape(-1, pof_size[1], pof_size[0]).astype(np.float)

        return np.nan_to_num(pof), pof_valid


class POF_Generator_cuda(nn.Module):
    def __init__(self,pof_size=[368, 368], pof_conns=np.array([[0, 1], [1, 2]]), k=100, dist_thresh=24.0, normalize=True, exploit_uvz=True, *args, **kargs):
        super(POF_Generator_cuda, self).__init__()
        dist_thresh = max(pof_size[1], pof_size[0]) / 368 * dist_thresh
        self.num_pof, self.pof_size = pof_conns.shape[0], pof_size
        x_m = np.tile(np.arange(pof_size[0]), (self.num_pof, pof_size[1], 1)).reshape(1, self.num_pof, pof_size[1], pof_size[0])
        y_m = np.tile(np.arange(pof_size[1]), (self.num_pof, pof_size[0], 1)).transpose([0, 2, 1]).reshape(1, self.num_pof, pof_size[1], pof_size[0])
        self.register_buffer('x_m',			torch.from_numpy(x_m).float())
        self.register_buffer('y_m',			torch.from_numpy(y_m).float())
        self.register_buffer('dist_thresh', torch.tensor(dist_thresh).float())
        self.register_buffer('pof_conns',	torch.from_numpy(pof_conns).long())
        self.register_buffer('scale',		torch.tensor(pof_size).float().reshape(1, 1, 2))
        self.normalize = normalize
        self.k = k
        self.exploit_uvz=exploit_uvz

    #key_points_2d batch x k x 3
    def forward(self, key_points_2d, key_points_3d, *args, **kargs):
        nb, nk = key_points_2d.shape[0:2]
        pt_xy, pt_valid_2d, pt_xyz, pt_valid_3d = key_points_2d[..., :2]*self.scale, key_points_2d[..., 2], key_points_3d[..., :3], key_points_3d[..., 3] < 1
        num_pof, pof_size, pof_conns = self.num_pof, self.pof_size, self.pof_conns
        pt_a_2d, pt_b_2d = pt_xy[:, pof_conns[:, 0]], pt_xy[:, pof_conns[:, 1]]
        vab = F.normalize(pt_b_2d - pt_a_2d, p=2, dim=2) # n x k x 2
        dx, dy = self.x_m - pt_a_2d[..., 0].reshape(nb, num_pof, 1, 1), self.y_m-pt_a_2d[..., 1].reshape(nb, num_pof, 1, 1)
        dist = torch.abs(dy*vab[..., 0].reshape(nb, num_pof, 1, 1)-dx*vab[..., 1].reshape(nb, num_pof, 1, 1))

        pt_x = torch.stack([pt_a_2d[...,0], pt_b_2d[...,0]], -1)
        pt_y = torch.stack([pt_a_2d[...,1], pt_b_2d[...,1]], -1)
        x_min, x_max = pt_x.min(-1)[0].reshape(nb, num_pof, 1, 1)-self.dist_thresh, pt_x.max(-1)[0].reshape(nb, num_pof, 1, 1)+self.dist_thresh
        y_min, y_max = pt_y.min(-1)[0].reshape(nb, num_pof, 1, 1)-self.dist_thresh, pt_y.max(-1)[0].reshape(nb, num_pof, 1, 1)+self.dist_thresh

        if not self.exploit_uvz:
            mask = ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[:, pof_conns[:, 0]]<3) * (pt_valid_2d[:, pof_conns[:, 1]]<3) * pt_valid_3d[:, pof_conns[:, 0]] * pt_valid_3d[:, pof_conns[:, 1]]).reshape(nb, -1, 1, 1)).float()
        else:
            mask = ((self.x_m >= x_min) * (self.x_m <= x_max) * (self.y_m >= y_min) * (self.y_m <= y_max) * (dist <= self.dist_thresh) * ((pt_valid_2d[:, pof_conns[:, 0]]<3) * (pt_valid_2d[:, pof_conns[:, 1]]<3)).reshape(nb, -1, 1, 1)).float()

        if self.normalize:
            vab_3d = F.normalize(input=pt_xyz[:, pof_conns[:, 1]]-pt_xyz[:, pof_conns[:, 0]], p=2, dim=2)
        else:
            vab_3d = pt_xyz[:, pof_conns[:, 1]]-pt_xyz[:, pof_conns[:, 0]]

        if not self.exploit_uvz:
            pof_value = vab_3d
        else:
            pof_value = torch.cat([vab, vab_3d[:,:,2:3]], 2)

        pof = mask.reshape(nb, num_pof, pof_size[1], pof_size[0], 1).expand(nb, num_pof, pof_size[1], pof_size[0], 3) *  pof_value.reshape(nb, num_pof, 1, 1, 3)
        pof = pof.permute(0, 1, 4, 2, 3).reshape(nb, num_pof*3, pof_size[1], pof_size[0])
    
        pt_valid_2d = ((pt_valid_2d[:, pof_conns[:, 0]]<3) * (pt_valid_2d[:, pof_conns[:, 1]]<3)).reshape(nb, -1, 1, 1).float()
        pt_valid_3d = (pt_valid_3d[:, pof_conns[:, 0]] * pt_valid_3d[:, pof_conns[:, 1]]).reshape(nb, -1, 1, 1).float()
        if not self.exploit_uvz:
            pof_valid = mask*pt_valid_3d*pt_valid_2d + pt_valid_3d*pt_valid_2d
            pof_valid = pof_valid.unsqueeze(-1).expand(nb, num_pof, pof_size[1], pof_size[0], 3).permute(0,1,4,2,3).reshape(nb, -1, pof_size[1], pof_size[0]).float()
        else:
            pof_valid_a = pt_valid_2d.reshape(nb, num_pof, 1, 1, 1).expand(nb, num_pof, pof_size[1], pof_size[0], 2)
            pof_valid_b = pt_valid_3d.reshape(nb, num_pof, 1, 1, 1).expand(nb, num_pof, pof_size[1], pof_size[0], 1)
            pof_valid = torch.cat([pof_valid_a, pof_valid_b], 4).permute(0, 1, 4, 2, 3).reshape(nb, -1, pof_size[1], pof_size[0])

        return pof, pof_valid

class HM_Generator(object):
    def __init__(self, hm_size=[368, 368], sigma=7, num_kp=18, *args, **kargs): #hm_size <=====> (w, h)
        sigma = max(hm_size[1], hm_size[0]) / 368 * sigma
        self.hm_size, self.sigma, self.num_kp = hm_size, sigma, num_kp
        x_m = np.tile(np.arange(hm_size[0]), (num_kp, hm_size[1], 1))
        y_m = np.tile(np.arange(hm_size[1]), (num_kp, hm_size[0], 1)).transpose([0, 2, 1])
        self.blk_hm = np.stack([x_m, y_m], -1) # num_kp x hm_size[1] x hm_size[0] x 2
        self.scale	= np.array(hm_size).reshape(1, 2)

    def __call__(self, key_points_2d, *args, **kargs): #key_points_2d is num_kp x 3, 3 <====> (x, y, conf)
        pt_xy, pt_valid = (key_points_2d[:, :2]*self.scale).reshape(self.num_kp, 1, 1, 2), key_points_2d[:, 2]
        dist = ((self.blk_hm-pt_xy)**2).sum(-1) # to num_kp x hm_size[1] x hm_size[0]
        hm = np.exp(-dist/(2*self.sigma*self.sigma)) * (pt_valid.reshape(self.num_kp, 1, 1) < 3).astype(np.float)
        hm_valid = (pt_valid < 3).astype(np.float)
        return hm, hm_valid

class HM_Generator_cuda(nn.Module):
    def __init__(self, hm_size=[368, 368], sigma=7, num_kp=21, *args, **kargs):
        super(HM_Generator_cuda, self).__init__()
        sigma = max(hm_size[1], hm_size[0]) / 368 * sigma
        self.hm_size, self.sigma, self.num_kp = hm_size, sigma, num_kp
        x_m = np.tile(np.arange(hm_size[0]), (num_kp, hm_size[1], 1))
        y_m = np.tile(np.arange(hm_size[1]), (num_kp, hm_size[0], 1)).transpose([0, 2, 1])
        blk_hm = np.stack([x_m, y_m], -1).reshape(1, num_kp, hm_size[1], hm_size[0], 2)
        self.register_buffer('blk_hm', torch.from_numpy(blk_hm).float())
        self.register_buffer('scale', torch.tensor(hm_size).float().reshape(1, 1, 2))			

    #inputs batch x self.num_kp x 3
    def forward(self, key_points_2d, *args, **kargs):
        nb = key_points_2d.shape[0]
        pt_xy, pt_valid = (key_points_2d[..., :2]*self.scale).reshape(nb, self.num_kp, 1, 1, 2), key_points_2d[..., 2]
        dist = ((self.blk_hm-pt_xy)**2).sum(-1) # to n x num_kp x hm_size[1] x hm_size[0]
        hm = torch.exp(-dist/(2*self.sigma*self.sigma)) *  (pt_valid.reshape(nb, self.num_kp, 1, 1) < 3) #if not in image, then return a heatmap with pure zeros
        return hm, (pt_valid < 3).float().reshape(nb, self.num_kp)