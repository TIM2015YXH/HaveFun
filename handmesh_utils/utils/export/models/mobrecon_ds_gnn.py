import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from mobhand.models.densestack import DenseStack_Backnone
from mobhand.models.modules import conv_layer, MMUpsample
import vctoolkit as vc
import numpy as np
from mobhand.tools.graph_utils import build_coarse_graphs, sparse_python_to_torch, build_graph
from mobhand.tools.coarsening import resymlaplacian
from conv.gcn import MMGCNLayer


class MMGCNDecode3D(nn.Module):
    def __init__(self, cfg, graph_info):
        super(MMGCNDecode3D, self).__init__()

        faces = graph_info['face'][::-1]
        up_transform = graph_info['up_transform'][::-1]
        self.graph_L = []
        for face in faces:
            mesh_adj = build_graph(face, face.max() + 1)
            mesh_adj.eliminate_zeros()
            L = resymlaplacian(mesh_adj, normalized=True)
            L = sparse_python_to_torch(L, dense=True)
            self.graph_L.append(L)

        self.out_channels = cfg.MODEL.GNN.OUT_CHANNELS
        self.heria_verts = [faces[i].max()+1 for i in range(len(faces))]
        self.faces = [torch.from_numpy(f.astype(np.int64)).permute(1, 0) for f in faces]
        layer = MMGCNLayer

        self.upsample = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(len(faces)):
            if i == 0:
                self.upsample.append(MMUpsample(faces[i].max()+1, 21))
            else:
                self.upsample.append(MMUpsample(faces[i].max()+1, faces[i-1].max()+1, m=torch.tensor(up_transform[i-1].todense()), learnable=cfg.MODEL.GNN.LEARN_UP))
            self.layers.append(layer(self.out_channels[i], self.out_channels[i+1], self.out_channels[i+1], bn=cfg.MODEL.GNN.BN))

        self.head = layer(self.out_channels[-1], self.out_channels[-1], 3, bn=False, relu=False)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        bs, c, h, w = x.size()
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.index(x, uv).permute(0, 2, 1)
        # x = x.view(bs, c, h*w).permute(0, 2, 1)

        for i in range(len(self.heria_verts)):
            x = self.upsample[i](x)
            x = self.layers[i](x, self.graph_L[i].to(x.device))

        x = self.head(x, self.graph_L[-1].to(x.device)).view(bs, self.heria_verts[-1], -1)

        return x


class MobRecon_DS_GNN(nn.Module):
    def __init__(self, cfg):
        super(MobRecon_DS_GNN, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
                                            kpts_num=cfg.MODEL.KPTS_NUM)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        transform_fp = os.path.join(cur_dir, '../../../template', 'transform.pkl')
        tmp = vc.load(transform_fp)
        decoder = MMGCNDecode3D
        self.decoder3d = decoder(cfg, tmp)

    def forward(self, x):
        latent, pred2d_pt = self.backbone(x)
        pred3d = self.decoder3d(pred2d_pt, latent)

        return pred3d, pred2d_pt, torch.tensor([1.0]).float()


if __name__ == '__main__':
    from mobhand.main import setup
    from options.cfg_options import CFGOptions
    args = CFGOptions().parse()
    args.config_file = 'mobhand/configs/mobrecon_ds_gnn.yml'
    cfg = setup(args)

    model = MobRecon_DS_GNN(cfg)
    print(
            "model created, param count: {}".format(
                sum([m.numel() for m in model.parameters() if m.requires_grad]) / 1e6
            )
        )
    
    model_out = model(torch.zeros(4, 3, 128, 128))
    print(model_out[0])
