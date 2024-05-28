#!/usr/bin/env python
# coding=utf-8

"""
Convert the pytorch model to onnx model.

Haoyanlong 2020-08-25
"""

import sys
sys.path.insert(0, '/home/chenxingyu/Documents/hand_mesh')

Checkpoint = "/home/chenxingyu/Documents/hand_mesh/lighthand/out/Kwai2D/cmrpng_reg2d_left_conm1cent_8gpu_fixbnparam_lr3/checkpoints/checkpoint_last.pt"
ONNXResFile = "./cmrpng_reg2d_left_conm1cent_8gpu_fixbnparam_lr3.onnx"
MLResFile = "./cmrpng_reg2d_left_conm1cent_8gpu_fixbnparam_lr3.mlmodel"

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import os
import cv2
import ipdb
import numpy as np

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx_coreml import convert
from utils.export.models.cmrpng_reg2d_left import CMRPNG_Reg2d_Left
from options.base_options import BaseOptions
from utils import spiral_tramsform

# from tf2pt.sa_gan import InpaintSANet
# from tf2pt.sa_gan_test import InpaintSANet

from torch.nn.parameter import Parameter
def load_state_dict(own_state,  new_state_dict):
    for name, param in new_state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                continue


class Pt2ONNXML:
    def __init__(self, checkpoint, onnx_resfile, ml_resfile):
        self.checkpoint = checkpoint
        self.onnx_resfile = onnx_resfile
        self.ml_resfile = ml_resfile

    def _load_pt_model(self):
        assert os.path.isfile(self.checkpoint), "The checkpoint must be existed!"
        args = BaseOptions().parse()
        args.seq_length = [9, 9, 9, 9]
        args.out_channels = [32, 64, 128, 256]
        args.work_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(args.work_dir, '../..', 'template', 'template.ply')
        transform_fp = os.path.join(args.work_dir, '../..', 'template', 'transform.pkl')
        spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation, None, 'cpu')
        for i in range(len(up_transform_list)):
            up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())
        model = CMRPNG_Reg2d_Left(args, spiral_indices_list, up_transform_list)
        model.eval()
        #weights = torch.load(self.checkpoint)
        #model.load_state_dict(weights)
    
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # load_state_dict(model.state_dict(), checkpoint['model'])

        imgs = torch.randn(1, 3, 128, 128)
        o = model(imgs)
        print (o[0].size(), o[1].size(), o[2].size())
        from thop import profile
        macs, params = profile(model, inputs=(imgs, ))
        print (macs/1000000000,params)
        return model

    def _valid_onnx_model(self):
        assert os.path.isfile(self.onnx_resfile), "The onnx resfile must be existed!"
        model = onnx.load(self.onnx_resfile)

        onnx.checker.check_model(model)
        onnx.helper.printable_graph(model.graph)

    def _onnx2ml(self):
        # Load the ONNX model as a CoreML model
        model = convert(model=self.onnx_resfile,
                        minimum_ios_deployment_target='13'
                        )

        # Save the CoreML model
        model.save(self.ml_resfile)


    def main(self):
        imgs = torch.randn(1, 3, 128, 128)
        # masks = torch.ones(1, 1, 256, 256)
        model = self._load_pt_model()

        print('Export ONNX model!')
        torch.onnx.export(model,
                          (imgs),
                          self.onnx_resfile,
                          verbose=True,
                          opset_version=11
                          # input_names=['input', 'mask'],
                          # output_names=['output']
                          )
        print('Valid ONNX model!')
        self._valid_onnx_model()

        print('Export CoreML model!')
        self._onnx2ml()


def grid_sampler(g, input, grid, mode, padding_mode, align_corners):  # long, long, long: contants dtype
    mode_i = sym_help._maybe_get_scalar(mode)
    paddingmode_i = sym_help._maybe_get_scalar(padding_mode)
    aligncorners_i = sym_help._maybe_get_scalar(align_corners)
    return g.op("myonnx_plugin::GridSampler", input, grid, interpolationmode_i=mode_i, paddingmode_i=paddingmode_i, aligncorners_i=aligncorners_i)  # just a dummy definition for onnx runtime since we don't need onnx inference

if __name__ == '__main__':
    # 自定义一个名为grid_sampler的OP
    import torch.onnx.symbolic_opset11 as sym_opset
    import torch.onnx.symbolic_helper as sym_help
    from torch.onnx import register_custom_op_symbolic

    # # 注册这个自定义的OP
    sym_opset.grid_sampler = grid_sampler
    register_custom_op_symbolic('myop::GridSampler', grid_sampler, 11)

    pt2onnxml = Pt2ONNXML(Checkpoint, ONNXResFile, MLResFile)
    pt2onnxml.main()
