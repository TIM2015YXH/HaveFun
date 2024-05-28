import sys
sys.path.insert(0, '/home/chenxingyu/Documents/hand_mesh')
import torch.nn as nn
import torch
from lighthand.models.modules_float import mobile_unit, conv_layer
from lighthand.models.modules import SpiralDeblock, SpiralConv_conv
from lighthand.models.cmrpng_reg2d import Backbone
import os
from mmcv.ops.point_sample import bilinear_grid_sample

class GridSampler(torch.nn.Module):
    def __init__(self):
        super(GridSampler, self).__init__()
    
    def forward(self, tenInput, g):
        return torch.nn.functional.grid_sample(input=tenInput, grid=g, align_corners=True)

class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel):
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel

        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1]))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1]))
        # self.gs = GridSampler()
        # head
        self.head = SpiralConv_conv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel])*0.01, requires_grad=True)


    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        # samples = self.gs(feat, uv)
        # samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        samples = bilinear_grid_sample(feat, uv, align_corners=True) 
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.index(x, uv).permute(0, 2, 1)
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)

        return pred

class CMRPNG_Reg2d_Left(nn.Module):
    def __init__(self, args, spiral_indices, up_transform):
        super(CMRPNG_Reg2d_Left, self).__init__()
        self.phase = args.phase
        self.uv_channel = 21
        self.input_channel = 128
        self.pose_loss_weight = args.pose_loss
        self.up_transform = up_transform
        self.pose_loss_weight = args.pose_loss
        self.backbone = Backbone(self.input_channel, 24, args.out_channels[-1])
        self.decoder3d = Reg2DDecode3D(self.input_channel * 2, args.out_channels, spiral_indices, up_transform, self.uv_channel)
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # weight = torch.load(os.path.join(cur_dir, '../out/PanoHand/cmrpng_reg2d_con2d_lr3/checkpoints/checkpoint_last.pt'), map_location='cpu')['model_state_dict']
        # self.load_state_dict(weight, strict=True)
        # self.eval()        
        # for param in self.parameters():
        #     param.requires_grad = False
        # print('Load pre-trained mesh weight and BN')


        self.mid_proj = nn.Sequential(conv_layer(256, 256, 3, 2, 1), conv_layer(256, 256, 1, 1, 0, bn=False, relu=False), 
                                      conv_layer(256, 256, 3, 2, 1), conv_layer(256, 256, 1, 1, 0, bn=False, relu=False))
        self.left_logits = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0), nn.ReLU(),
                                         nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(),
                                         nn.Conv2d(128, 64, 1, 1, 0), nn.ReLU(),
                                         )
        self.left_head = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        latent, pred2d_pt = self.backbone(x)
        mid_proj = self.mid_proj(latent)
        left_logits = self.left_logits(mid_proj)
        left_pred = self.left_head(left_logits).view(latent.size(0))
        pred3d = self.decoder3d(pred2d_pt, latent)
        return pred3d, pred2d_pt, left_pred



if __name__ == '__main__':
    import os.path as osp
    from utils import utils, writer, spiral_tramsform
    from options.base_options import BaseOptions
    from utils.flops import count_meshnet
    import cv2
    import numpy as np

    args = BaseOptions().parse()
    args.out_channels = [32, 64, 128, 256]
    args.size = 128
    args.phase = 'test'

    template_fp = osp.join('template', 'template.ply')
    transform_fp = osp.join('template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, [2, 2, 2, 2], [9, 9, 9, 9], [1, 1, 1, 1], '', '')
    for i in range(len(up_transform_list)):
        up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())


    model = CMRPNG_Reg2d_Left(args, spiral_indices_list, up_transform_list)
    model_path = '/home/chenxingyu/Documents/hand_mesh/lighthand/out/Kwai2D/cmrpng_reg2d_left_conm1cent_8gpu_fixbnparam_lr3/checkpoints/checkpoint_last.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(sum([m.numel() for m in model.parameters() if m.requires_grad]) / 1e6 )
    # count_meshnet(model, args.size)
    img = cv2.imread('/home/chenxingyu/Documents/cmr_demo_pytorch/images/119_img.jpg')[..., ::-1]
    img = cv2.resize(img, (128, 128)).astype(np.float32)
    img = (torch.from_numpy(img) / 255 - 0.5) / 0.5
    data = img.unsqueeze(0).permute(0, 3, 1, 2)
    # data = torch.zeros([1, 3, 128, 128])
    res = model(data)
    print(res[0], res[1], res[2])
