import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

# from .projection import ProjectXYZToUVD

from dmdrive.model.handy import Handy
from dmdrive.model.utils import make_aligned, to_homogeneous,batch_rodrigues

from dmdrive.smplx_model.smplx import SMPLX

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None
        if opt.uvd_proj:
            # self.converter = ProjectXYZToUVD()
            print('use uvd projection!!!!!!!!!!!')
        else:
            self.converter = None

        if self.opt.real_mesh_scale_path:
            self.mesh_scale = np.load(self.opt.real_mesh_scale_path)
        else:
            self.mesh_scale = 1
        
        if self.opt.real_shape_scale:
            self.real_shape_scale = np.load(self.opt.real_shape_scale).item()
        else:
            self.real_shape_scale = 1.

        self.mano_model = self.handy_model = self.smplx_model = None
        if opt.handy_path is not None:
            self.handy_model = Handy(opt.handy_path, device=torch.device('cuda'))    
        elif opt.smplx_path is not None:
            self.smplx_model = SMPLX(opt.smplx_path, gender=opt.gender, device=torch.device('cuda'))    
        elif opt.mano_path is not None:
            from smplx import build_layer
            mano_cfgs = {
                'model_folder': opt.mano_path,
                'model_type': 'mano',
                'num_betas': 10
            }
            self.mano_model = build_layer(
                        mano_cfgs['model_folder'], model_type = mano_cfgs['model_type'],
                        num_betas = mano_cfgs['num_betas']
                    ).cuda()
            
            # zero_pose = torch.zeros(48).to(torch.float32).cuda()
            # zero_shape = torch.zeros(10).to(torch.float32).cuda()
            # theta_rodrigues = batch_rodrigues(zero_pose.reshape(-1, 3)).reshape(1, 16, 3, 3)
            # __theta = theta_rodrigues.reshape(1, 16, 3, 3)
            # so = self.mano_model(betas = zero_shape.reshape(1, 10), hand_pose = __theta[:, 1:], global_orient = __theta[:, 0].view(1, 1, 3, 3))
            # v_rest = so['vertices'].clone().reshape(1, -1, 3)
            # joints = so['joints'].clone().reshape(-1,3)
            # faces_rest = self.mano_model.faces # [f, 3]
            # self.faces_rest = faces_rest.astype(np.int32)
            # v_rest = v_rest[0]
            # self.joints_rest = joints
            # self.mat_aligned = make_aligned(self.joints_rest, self.mesh_scale) # including minus joint4
            # self.v_rest = torch.einsum('Ni, Bi->BN', self.mat_aligned, to_homogeneous(v_rest))[:,:3]
            # self.joints_rest_aligned = torch.einsum('Ni, Bi->BN', self.mat_aligned, to_homogeneous(self.joints_rest))[:,:3]
            

    def common_forward(self, x):

        # sigma
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)

        h = self.sigma_net(enc)

        if self.converter is not None:
            sigma = self.density_activation(h[..., 0])
        else:
            sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo = self.common_forward(x)
        # normal = None
        # if not self.training:
            # normal = self.normal(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading

            # normal = self.normal_net(enc)
            # if normal is None:
            normal = self.normal(x)

            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            # {'params': self.normal_net.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})

        return params