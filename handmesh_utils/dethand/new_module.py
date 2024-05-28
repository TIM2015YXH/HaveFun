import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class AvgQuant(nn.Module):
    dump_patches = True
    def __init__(self):
        super(AvgQuant, self).__init__()
    def forward(self, xs):
        x1, x2 = xs
        out = (x1 + x2)/2
        out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out

class AdaptiveAvg(nn.Module):
    dump_patches = True
    def __init__(self, weights=None):
        super(AdaptiveAvg, self).__init__()
        self.weights = weights
    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            if self.weights is None:
                self.weights = [1.0/len(list(x)) for i in range(len(list(x)))]
            out = x[0] * self.weights[0]
            for i in range(1, len(x)):
                out += x[i] * self.weights[i]
        else:
            raise ValueError('the type of weights is error.')
        return out

class AdaptiveAvgQuant(nn.Module):
    dump_patches = True
    def __init__(self, weights=None):
        super(AdaptiveAvgQuant, self).__init__()
        self.weights = weights
    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            if self.weights is None:
                self.weights = [1.0/len(list(x)) for i in range(len(list(x)))]
            out = x[0] * self.weights[0]
            for i in range(1, len(x)):
                out += x[i] * self.weights[i]
        else:
            raise ValueError('the inputs of AdaptiveAvgQuant must be list/tuple.')
        out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out

class HardQuant(nn.Module):
    dump_patches = True
    def __init__(self, min_val = 0, max_val = 4, inplace = False):
        super(HardQuant, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.delta = self.max_val - self.min_val
        self.inplace = inplace
    def forward(self, x):
        out = F.hardtanh(x, self.min_val, self.max_val, self.inplace)
        if self.min_val == 0:  # 只有最小为0时，需要量化，否则输出是浮点
            out.data[:] = torch.round(out.data[:] * (255. / self.delta)) * (self.delta / 255.)
        return out

# deprecated: same with HardQuant above
class QuantizedHardtanh(nn.Module):
    dump_patches = True
    def __init__(self, min_val = 0, max_val = 4, inplace = False):
        super(QuantizedHardtanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.delta = self.max_val - self.min_val
        self.inplace = inplace
    def forward(self, x):
        out = F.hardtanh(x, self.min_val, self.max_val, self.inplace)
        out.data[:] = torch.round(out.data[:] * (255. / self.delta)) * (self.delta / 255.)
        return out

# Quant Model should use this. Float model just use * directly
class SliceMul(nn.Module):
    dump_patches = True
    """
    x1: NxCxHxW feature map
    x2: NxCx1x1 SE layer output
    """
    def __init__(self):
        super(SliceMul, self).__init__()

    def forward(self, x1, x2):
        out = x1 * x2
        # Note: divide by 256, consistance with Modile-Neon
        out.data[:] = torch.floor(out.data[:] * (255. / 256.) * (255. / 4.)) * (4. / 255.)
        return out

# Quant Model should use this. Float model just use * directly
class SliceMulParams(nn.Module):
    dump_patches = True
    """
    x1: NxCxHxW feature map
    x2: NxCx1x1 SE layer output
    """
    def __init__(self, out_channels, weights):
        super(SliceMulParams, self).__init__()
        self.out_channels = out_channels
        self.weights = torch.from_numpy(weights.astype(np.float32).reshape(1, out_channels, 1, 1))

    def forward(self, x1):
        out = x1 * self.weights.expand(-1, self.out_channels, 1, 1)
        return out

# Quant Model should use this. Float model just use * directly
class SliceAdd(nn.Module):
    dump_patches = True
    """
    x1: NxCxHxW feature map
    x2: NxCx1x1 SE layer output
    """
    def __init__(self, out_channels, beta):
        super(SliceAdd, self).__init__()
        self.out_channels = out_channels
        self.weights = torch.from_numpy(beta.astype(np.float32).reshape(1, out_channels, 1, 1))

    def forward(self, x1, x2):
        out = x1 + self.weights.expand(-1, self.out_channels, 1, 1)
        return out

# deprecated: MulQuant should not used directly in Quant Models
class MulQuant(nn.Module):
    dump_patches = True
    def __init__(self):
        super(MulQuant, self).__init__()

    def forward(self, x1, x2):
        out = x1 * x2
        # divide by 256, consistance with Modile-Neon
        out = torch.clamp(out, 0, 4)
        out.data[:] = torch.floor(out.data[:] * (255. / 256.) * (255. / 4.)) * (4. / 255.)
        return out

# Consider to use AvgQuant in Quant Models
class AddQuant(nn.Module):
    dump_patches = True
    def __init__(self, max_val=4.0):
        super(AddQuant, self).__init__()
        self.max_val = max_val
    def forward(self, x1, x2):
        out = x1 + x2
        out = torch.clamp(out, 0, self.max_val)
        return out

class AvgPoolQuant(nn.AvgPool2d):
    dump_patches = True
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPoolQuant, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                      count_include_pad=count_include_pad)

    def forward(self, x):
        #x = torch.round(x.data[:] * (255. / 4.))  # TODO：测试有效性
        x = torch.round(x * (255. / 4.))
        out = F.avg_pool2d(x, self.kernel_size, self.stride,
                           self.padding, self.ceil_mode, self.count_include_pad)
        #out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        out = torch.floor(out) * (4. / 255.)
        #out.data[:] = torch.floor(out.data[:]) * (4. / 255.)
        return out

class AdaptiveAvgPoolQuant(nn.AdaptiveAvgPool2d):
    dump_patches = True
    def __init__(self, output_size):
        super(AdaptiveAvgPoolQuant, self).__init__(output_size)
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, self.output_size)
        out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out

class UpsampleQuant(nn.Upsample):
    dump_patches = True
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(UpsampleQuant, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.maxv = 4

    def forward(self, x):
        if hasattr(self, 'align_corners'):
            if self.mode == "nearest":
                out = F.upsample(x, self.size, self.scale_factor, self.mode)
            else:
                out = F.upsample(x, self.size, self.scale_factor, self.mode, True)
        else:
            out = F.upsample(x, self.size, self.scale_factor, self.mode)
        out.data[:] = torch.round(out.data[:] * (255. / self.maxv)) * (self.maxv / 255.)
        return out

class Conv2dQuant(nn.Conv2d):
    dump_patches = True
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)
        self.register_buffer('weight_copy', torch.zeros(self.weight.size()))
        self.computation = 0

    def get_computation(self, x):
        feature_size = x.size()[-1]
        return (feature_size) ** 2 * self.kernel_size[0] ** 2 * self.in_channels * self.out_channels / (self.groups * 2 ** 20)

    def forward(self, input):
        # self.computation = self.get_computation(input)
        self.weight_copy[:] = self.weight.data[:]
        weight_max = torch.abs(self.weight_copy).max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        self.weight.data[:] = torch.round((self.weight.data[:] / (weight_max  + 1e-12)) * 127) * weight_max / 127
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.weight.data[:] = self.weight_copy[:]
        return out

class ConvTranspose2dQuant(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        self.register_buffer('weight_copy', torch.zeros(self.weight.size()))
    def forward(self, input, output_size=None):
        self.weight_copy[:] = self.weight.data[:]
        self.weight.data = self.weight.data.permute(1, 0, 2, 3)
        groups = self.groups
        # cgl -> channelin_group_len
        cgl = int(self.in_channels / groups)
        for i in range(groups):
            subarray = self.weight.data[:,i * cgl: (i + 1) * cgl]
            subarray_max = torch.abs(subarray).max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            self.weight.data[:, i * cgl: (i + 1) * cgl] = torch.round((self.weight.data[:, i * cgl: (i + 1) * cgl] / (subarray_max + 1e-12)) * 127) * subarray_max / 127
        self.weight.data = self.weight.data.permute(1, 0, 2, 3)
        out = F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        self.weight.data[:] = self.weight_copy[:]
        return out

class LinearQuant(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(LinearQuant, self).__init__(in_features, out_features, bias=bias)
        self.register_buffer('weight_copy', torch.zeros(self.weight.size()))

    def forward(self, input):
        self.weight_copy[:] = self.weight.data[:]
        weight_max = \
            torch.abs(self.weight_copy).max(dim=1, keepdim=True)[0]
        self.weight.data[:] = torch.round((self.weight.data[:] / (weight_max  + 1e-12)) * 127) * weight_max / 127
        out = F.linear(input, self.weight, self.bias)
        self.weight.data[:] = self.weight_copy[:]
        return out

class Swish(nn.Module):
    dump_patches = True
    def __init__(self):
        super(Swish, self).__init__()
        self.register_buffer('k', torch.zeros(12))
        self.register_buffer('b', torch.zeros(12))
        self.exp_points()
    def forward(self, x):
        x = F.hardtanh(x, -4, 4)
        x_ = x.unsqueeze(-1)
        x_seg = (-x_)*Variable(self.k, requires_grad=False)+ Variable(self.b, requires_grad=False)
        x_exp, _ = torch.max(x_seg, dim=-1)
        out = x/(1 + x_exp) + 0.2785
        out = F.hardtanh(out, -4, 4, inplace=False)
        out.data[:] = torch.round(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out
    def exp_points(self):
        t = np.arange(-6., 7., 1.)
        y_p = 2.**t
        y_p_diff = y_p[1:] - y_p[:-1]
        b = y_p[1:] - y_p_diff*t[1:]
        k = y_p_diff/np.log(2)
        self.k, self.b = (torch.from_numpy(k)).float(), \
                         (torch.from_numpy(b)).float()

# 若训练时采用nn.Sigmoid不近似e指数版本，将导出为type 60(SigmoidAccurate)
class SigmoidQuant(nn.Module):
    def __init__(self):
        super(SigmoidQuant, self).__init__()
        self.register_buffer('k', torch.zeros(12))
        self.register_buffer('b', torch.zeros(12))
        self.exp_points()
    def forward(self, x):
        x = F.hardtanh(x, -4, 4)
        x_ = x.unsqueeze(-1)
        x_seg = (-x_) * Variable(self.k, requires_grad=False) + Variable(self.b, requires_grad=False)
        x_exp, _ = torch.max(x_seg, dim=-1)
        out = 1/(1 + x_exp)
        out.data[:] = torch.round(out.data[:] * (255. / 1.)) * (1. / 255.)
        return out
    def exp_points(self):
        t = np.arange(-6., 7., 1.)
        y_p = 2.**t
        y_p_diff = y_p[1:] - y_p[:-1]
        b = y_p[1:] - y_p_diff*t[1:]
        k = y_p_diff/np.log(2)
        self.k, self.b = (torch.from_numpy(k)).float(), (torch.from_numpy(b)).float()

# 近似e指数，与ycnn底层实现相同
class ExpAprox(nn.Module):
    def __init__(self):
        super(ExpAprox, self).__init__()
    k = torch.zeros(12)
    b = torch.zeros(12)
    t = np.arange(-6., 7., 1.)
    y_p = 2. ** t
    y_p_diff = y_p[1:] - y_p[:-1]
    bb = y_p[1:] - y_p_diff * t[1:]
    kk = y_p_diff / np.log(2)
    k, b = (torch.from_numpy(kk)).float(), (torch.from_numpy(bb)).float()

    def forward(self, x):
        x = F.hardtanh(x, -4, 4)
        x_ = x.unsqueeze(-1)
        x_seg = (x_) * Variable(self.k, requires_grad=False) + Variable(self.b, requires_grad=False)
        x_exp, _ = torch.max(x_seg, dim=-1)
        return x_exp

class NormSigmoid(nn.Module):
    def __init__(self):
        super(NormSigmoid, self).__init__()

    def forward(self, x):
        out = torch.sigmoid(x) / torch.sigmoid(x).sum(dim=1).unsqueeze(1)
        return out

# deprecated: 浮点模块，不带Quant, 直接用AvgPool2d就可以
class AvgPool(nn.AvgPool2d):
    dump_patches = True

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                      count_include_pad=count_include_pad)

    def forward(self, x):
        out = F.avg_pool2d(x, self.kernel_size, self.stride,
                           self.padding, self.ceil_mode, self.count_include_pad)
        # out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out

class LeakyReLU(nn.Module):
    def __init__(self, val=1e-2):
        super(LeakyReLU, self).__init__()
        self.relu = nn.ReLU()
        self.val = val

    def forward(self, x):
        return x * self.val + self.relu(x) * (1-self.val)

### 一些特殊的Module封装 ###
# set quantize=True if use Quantized model
class RoiAlign(nn.Module):
    dump_patches = True
    def __init__(self, input_size, output_size, min_val=0, max_val=4, quantize=True):
        super(RoiAlign, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.min_val = min_val
        self.max_val = max_val
        self.delta = self.max_val - self.min_val
        self.quantize = quantize

    def forward(self, x):
        input, bbox = x
        bbox = bbox.numpy().astype(float)
        x = torch.linspace(0, self.output_size[0] - 1, self.output_size[0])
        y = torch.linspace(0, self.output_size[1] - 1, self.output_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grids = x_grid.unsqueeze(0).repeat(input.size(0), 1, 1).numpy()
        y_grids = y_grid.unsqueeze(0).repeat(input.size(0), 1, 1).numpy()
        x_grids = (((x_grids * (bbox[:, 2] - bbox[:, 0]) + bbox[:, 0]) -0.5) * 2)
        y_grids = (((y_grids * (bbox[:, 3] - bbox[:, 1]) + bbox[:, 1]) -0.5) * 2)
        x_grids = np.expand_dims(x_grids, axis=3)
        y_grids = np.expand_dims(y_grids, axis=3)
        grids = np.concatenate((x_grids, y_grids), axis=3)
        grids = torch.Tensor(grids)
        roi = F.grid_sample(input, grids)
        if self.quantize:
            roi = torch.round(roi * (255. / self.delta)) * (self.delta / 255.)
        return roi

class H_Sigmoid(nn.Module):
    def __init__(self):
        super(H_Sigmoid, self).__init__()

    def forward(self, x):
        out = F.hardtanh(x + 3, 0, 6) / 6
        return out

# 此op，后面不可加cat，add，avg
class H_SigmoidQuant(nn.Module):
    def __init__(self, min_val=0, max_val=1, inplace=False):
        super(H_SigmoidQuant, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.delta = self.max_val - self.min_val
        self.inplace = inplace

    def forward(self, x):
        out = F.hardtanh(x + 3, 0, 6) / 6
        if self.min_val == 0:  # 只有最小为0时，需要量化，否则输出是浮点
            out.data[:] = torch.round(out.data[:] * (255. / self.delta)) * (self.delta / 255.)
        else:
            raise Exception("只有最小为0时，需要量化，否则输出是浮点")
        return out

class H_Swish(nn.Module):
    def __init__(self):
        super(H_Swish, self).__init__()

    def forward(self, x):
        out = x * F.hardtanh(x + 3, 0, 6) / 6
        return out

class H_SwishQuant(nn.Module):
    def __init__(self, min_val = 0, max_val = 4, inplace = False):
        super(H_SwishQuant, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.delta = self.max_val - self.min_val
        self.inplace = inplace

    def forward(self, x):
        out = F.hardtanh(x, self.min_val, self.max_val, self.inplace)
        out = out * F.hardtanh(out + 3, 0, 6) / 6
        if self.min_val == 0:  # 只有最小为0时，需要量化，否则输出是浮点
            out.data[:] = torch.round(out.data[:] * (255. / self.delta)) * (self.delta / 255.)
        else:
            raise Exception("只有最小为0时，需要量化，否则输出是浮点")
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, unbiased=True):
        super(ILN, self).__init__()
        self.eps = eps
        self.unbiased = unbiased
        self.num_features = num_features
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        # in_mean, in_var = torch.mean(torch.mean(input, dim=2, keepdim=True), dim=3, keepdim=True), torch.var(torch.var(input, dim=2, keepdim=True), dim=3, keepdim=True)
        # in_mean, in_var = torch.mean(input, dim=[2,3], keepdim=True), torch.var(input, dim=[2,3], keepdim=True)
        n, c, h, w = input.shape
        input_in = input.view(n, c, -1)
        in_mean, in_var = torch.mean(input_in, dim=2, keepdim=True).unsqueeze(-1), torch.var(input_in, dim=2, keepdim=True, unbiased=self.unbiased).unsqueeze(-1)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)

        # ln_mean, ln_var = torch.mean(torch.mean(torch.mean(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True), torch.var(torch.var(torch.var(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        # ln_mean, ln_var = torch.mean(input, dim=[1,2,3], keepdim=True), torch.var(input, dim=[1,2,3], keepdim=True)
        input_ln = input.view(n, -1)
        ln_mean, ln_var = torch.mean(input_ln, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1), torch.var(input_ln, dim=1, keepdim=True, unbiased=self.unbiased).unsqueeze(-1).unsqueeze(-1)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        self.rho.data = self.rho.data.clamp(0.0, 1.0)

        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class WeightCenter(nn.Module):
    dump_patches = True
    def __init__(self, height, width):
        super(WeightCenter, self).__init__()
        self.height = height
        self.width = width
        pos_x, pos_y = np.meshgrid(
            np.linspace(0., 1., self.height),
            np.linspace(0., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()  # TODO: reshape of not according to .view
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, x):
        expected_x = torch.sum(Variable(self.pos_x, requires_grad=False) * x, dim=2)
        expected_y = torch.sum(Variable(self.pos_y, requires_grad=False) * x, dim=2)
        expected_x_numpy = expected_x.cpu().numpy()
        expected_y_numpy = expected_y.cpu().numpy()
        expected_xy_numpy = np.concatenate((expected_x_numpy, expected_y_numpy), axis=1)
        expected_xy = torch.from_numpy(expected_xy_numpy)
        if torch.cuda.is_available():
            expected_xy = expected_xy.cuda()
        expected_xy = Variable(expected_xy, requires_grad=False)
        return expected_xy

class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2).\
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, cut=True, relu=True,
               group=1, weight_quant=True):
    if weight_quant:
        _conv = Conv2dQuant
    else:
        _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        if cut:
            sequence.append(HardQuant(0, 4))
        else:
            sequence.append(nn.ReLU())
    else:
        if cut:  # TODO
            pass#sequence.append(HardQuant(0, 4))
        else:
            print("Warning! No activation layer followed after Conv!")
    print('computation:', channel_in, channel_out, ks, stride)
    return nn.Sequential(*sequence)

def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True, unit=False, weight_quant=True):
    if not unit:
        if weight_quant:
            _linear = LinearQuant
        else:
            _linear = nn.Linear
        sequence = [_linear(channel_in, channel_out, bias=bias)]
    else:
        sequence = [LinearUnit(channel_in, channel_out, bias=bias)]
    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(HardQuant(0, 4))
    return nn.Sequential(*sequence)

class LinearUnit(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearUnit, self).__init__(in_features, out_features, bias=bias)
        self.weight_unit = None
    def forward(self, input):
        weight_norm = torch.sqrt((self.weight ** 2).sum(dim=1, keepdim=True))
        self.weight_unit = self.weight / weight_norm
        out = F.linear(input, self.weight_unit, self.bias)
        return out

class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, num3x3=1, has_half_out=False):
        print('Making mobile_unit!')
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out
        self.avg = AvgQuant()

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = self.avg(out, x)
        if self.has_half_out:
            return half_out, out
        else:
            return out

class mobile_unit_v2(nn.Module):
    dump_patches = True
    def __init__(self, channel_in, channel_out, stride=1, num3x3=1):
        print('Making mobile_unit_v2!')
        super(mobile_unit_v2, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if self.stride > 1 or self.channel_in < self.channel_out:
            channel_3x3 = channel_in
        else:
            channel_3x3 = channel_out

        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_3x3, channel_3x3, ks=3, stride=stride, padding=1, group=channel_3x3),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_3x3, channel_3x3, ks=3, stride=1, padding=1, group=channel_3x3),
                conv_layer(channel_3x3, channel_3x3, ks=3, stride=stride, padding=1, group=channel_3x3),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.avg = AvgQuant()

    def forward(self, x):
        if self.stride > 1 or self.channel_in < self.channel_out:
            half_out = self.conv3x3(x)
            out = self.conv1x1(half_out)
        else:
            out = self.conv1x1(x)
            out = self.conv3x3(out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = self.avg(out, x)
        return out

class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit_v2(channel_in, channel_in//4)
        self.conv2 = mobile_unit_v2(channel_in*5//4, channel_in//4)
        self.conv3 = mobile_unit_v2(channel_in*6//4, channel_in//4)
        self.conv4 = mobile_unit_v2(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), 1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), 1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3), 1)
        out4 = self.conv4(comb3)
        comb4 = torch.cat((comb3, out4), 1)
        return comb4

class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit_v2(channel_in, channel_in//2)
        self.conv2 = mobile_unit_v2(channel_in*3//2, channel_in//2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), 1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), 1)
        return comb2

def wing_loss(x, t, omega, sigma):
    c = omega - omega * np.log(1 + omega / sigma)
    diff_abs = torch.abs(x - t)
    small_loss = omega * torch.log(1 + diff_abs / sigma)
    big_loss = diff_abs - c
    is_small = (diff_abs < omega).float()
    return torch.mean(small_loss * is_small + big_loss * (1 - is_small))

### 以下class为导模型内部使用标记，不作为训练使用 ###
class ElementwiseAdd():
    def __init__(self):
        pass

class ElementwiseSub():
    def __int__(self):
        pass

class MulConstant():
    def __init__(self):
        pass

class Reshape():
    def __init__(self):
        pass

class View():
    def __init__(self):
        pass

class Mul():
    def __init__(self):
        pass

class SubConstant():
    def __init__(self):
        pass

class Concat():
    def __init__(self):
        pass

class Transpose():
    def __init__(self):
        pass

class BMM():
    def __init__(self):
        pass

class Sum():
    def __init__(self):
        pass

class SliceDiv():
    def __init__(self):
        pass

class SlicePickRange():
    def __init__(self):
        pass

class DirectCopy():
    def __int__(self):
        pass

class RNNNode:
    def __init__(self, id = -1, input_idx = -1, output_idx = -1, is_quantize = False, value = None):
        self.id = id
        self.input_index = input_idx if id != 65535 else 65535
        self.output_index = output_idx
        self.type = "fixed" if is_quantize else "float"
        self.shape = value.shape
        self.value = value
        self.is_quantize = is_quantize

class RNNRecord:
    def __init__(self, start_index = -1, inp_dict = OrderedDict()):
        self.start_idx = start_index
        self.rnn_inp_dict = inp_dict
        self.rnn_inp_idxs = []
        self.rnn_inp_idxs_replaced = []
        self.rnn_out_idxs = []
        if len(self.rnn_inp_dict.keys()) > 0:
            self.rnn_inp_idxs = [self.rnn_inp_dict[item].id for item in self.rnn_inp_dict]
            print("Rnn input idxes: ", self.rnn_inp_idxs)
            self.rnn_inp_idxs_replaced = [-i for i in range(len(self.rnn_inp_idxs))]
            if self.rnn_inp_idxs[0] == 65535:
                self.rnn_inp_idxs_replaced[0] = 65535
            self.rnn_out_idxs = [[] for i in range(len(self.rnn_inp_dict.keys()))]

# 该模块是为了兼容之前的旧模型
# 模型中使用torch.cat()，不要使用本Cat
class Cat(nn.Module):
    dump_patches = True
    def __init__(self):
        super(Cat, self).__init__()
        self.__name__ = "Cat"
    def forward(self, x):
        return torch.cat(x, dim=1)
    # For RNN model
    #def forward(self, x0, x1):
    #    return torch.cat([x0, x1], dim=1)

class AdaptiveAvgPool(nn.AdaptiveAvgPool2d):
    dump_patches = True
    def __init__(self, output_size):
        super(AdaptiveAvgPool, self).__init__(output_size)
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, self.output_size)
        out.data[:] = torch.floor(out.data[:] * (255. / 4.)) * (4. / 255.)
        return out

class RCConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(RCConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)
        self.enable_mac_cnt = False
        self.mac_cnt = None
        self.mac_cnt_bias = None
        # in_bn, out_bn layers, None means to be filled, 1 means to be copied (skip connection)
        self.in_out_bn = [None, None]
        self._wn = False
        self.prev_conv = []

    @property
    def wn(self):
        return self._wn

    @wn.setter
    def wn(self, val):
        if self.in_out_bn[0] is None:
            raise ValueError
        in_bn = self.in_out_bn[0]
        if isinstance(in_bn, list):
            self.id_in_channels = sum([bn.weight.numel() for bn in in_bn])
        else:
            self.id_in_channels = in_bn.weight.numel()
        assert self.id_in_channels == self.in_channels or self.id_in_channels * 4 == self.in_channels
        if self.groups == 1:
            channel_norms = self.weight.data.transpose(0, 1).contiguous().view(self.id_in_channels, -1).norm(p=2, dim=-1).clamp_(min=1e-12)
        else:
            channel_norms = self.weight.data.view(self.id_in_channels, -1).norm(p=2, dim=-1).clamp_(min=1e-12)
        if val:
            if isinstance(in_bn, list):
                i = 0
                for bn in in_bn:
                    bn.weight.data *= channel_norms[i:i+bn.weight.numel()]
                    bn.bias.data *= channel_norms[i:i+bn.bias.numel()]
                    i += bn.weight.numel()
            else:
                in_bn.weight.data *= channel_norms
                in_bn.bias.data *= channel_norms
        else:
            self.weight.data /= channel_norms.view(1, -1, 1, 1)
        self._wn = val

    def set_in_bn(self, bn_layer):
        self.in_out_bn[0] = bn_layer

    def get_in_bn(self):
        return self.in_out_bn[0]

    def set_out_bn(self, bn_layer):
        self.in_out_bn[1] = bn_layer

    def get_out_bn(self):
        return self.in_out_bn[1]

    def set_pre_conv(self, prev_conv_layers):
        self.prev_conv = prev_conv_layers

    def forward(self, input):
        if self.enable_mac_cnt:
            fake_output = F.conv2d(torch.ones_like(input.data), torch.ones_like(self.weight.data), None,
                                   self.stride, self.padding, self.dilation, self.groups)
            self.mac_cnt_bias = float(fake_output.numel())
            self.mac_cnt = fake_output.sum().item()
        if self.wn:
            if self.groups == 1:
                channel_norms = self.weight.data.transpose(0, 1).view(self.id_in_channels, -1).norm(p=2, dim=-1).clamp_(min=1e-12)
            else:
                channel_norms = self.weight.data.view(self.id_in_channels, -1).norm(p=2, dim=-1).clamp_(min=1e-12)
            weight = (self.weight.view(self.weight.size(0), channel_norms.numel(), -1) / channel_norms.view(1, -1, 1)).view(self.weight.shape)
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


LAYER_TYPE_NUM = {nn.Conv2d: 38, Conv2dQuant: 9, nn.LogSoftmax:34, nn.Softmax: 34, nn.ConvTranspose2d: 5, nn.MaxPool2d: 7, nn.BatchNorm2d: 8,
                  nn.BatchNorm1d: 8, nn.ReLU: 21, nn.Upsample: 13, UpsampleQuant:13, ElementwiseAdd: 18, nn.Hardtanh: 40, nn.Conv1d: 51, nn.MaxPool1d: 52,
                  MulConstant: 11, Concat: 12, HardQuant: 40, QuantizedHardtanh: 40, AvgQuant: 29, nn.Linear: 25, nn.Embedding: 62, nn.ConstantPad1d: 63,
                  Reshape: 26, nn.AvgPool2d: 15, AvgPoolQuant: 15, AvgPool: 15, nn.Sigmoid: 60, SigmoidQuant: 16, SliceMul: 261, Mul: 19, SubConstant: 45,
                  ElementwiseSub:43, nn.GRU: 49, nn.GRUCell: 50, nn.Tanh: 55, BMM: 81, Transpose: -1, Reorg: 28, AdaptiveAvgPoolQuant: 41, Sum: 278,
                  SliceDiv: 279, WeightCenter: 280, SlicePickRange: 270, LinearQuant: 30, nn.PixelShuffle: 67, nn.AdaptiveAvgPool2d: 41, nn.LSTM: 58,
                  nn.LSTMCell: 57, nn.LeakyReLU:69, View: 65, DirectCopy: 44, AddQuant: 18, ShuffleBlock: 287, nn.InstanceNorm2d: 56, ConvTranspose2dQuant: 31,
                  RoiAlign : 72, RCConv2d: 38, nn.ReflectionPad2d: 87, ILN: 288, H_Swish: 70, H_SwishQuant: 70, H_Sigmoid: 71, H_SigmoidQuant: 71,
                  AdaptiveAvgQuant: 289, SliceMulParams: 269, SliceAdd: 271}

SKIP_LAYERS = [Transpose]
### 以上为导模型内部使用标记，不作为训练使用 ###
version = "matting_v1_0"
key1 = [0x24, 0x8e, 0x27, 0x36, 0x61, 0x7e, 0xd3, 0x56, 0xa1, 0xf7, 0x15, 0x88, 0x19, 0xcf, 0x5f, 0x3c]
ekey = '248e2736617ed356a1f7158819cf5f3c'

flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list or type(x) is tuple else [x]

