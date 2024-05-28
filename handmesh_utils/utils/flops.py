from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
import pdb
#from new_modules import Cat

count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    ### ops_conv
    if type_name in ['Conv2d', 'Conv2dQuant']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        if x.size(0) > 100:
            delta_ops = delta_ops * x.size(0) / 2
        # print(layer)
        # print(delta_ops/ 1e6)
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        #delta_ops = x.numel()
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d', 'AvgPoolQuant']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        #delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d', 'AdaptiveAvgPool', 'AdaptiveAvgPoolQuant']:
        #delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear', 'LinearQuant']:
        weight_ops = layer.weight.numel() * multi_add

        #bias_ops = layer.bias.numel()
        #delta_ops = x.size()[0] * (weight_ops + bias_ops)
        if x.ndim == 2:
            delta_ops = weight_ops
        elif x.ndim == 3:
            delta_ops = x.size(1) * weight_ops
        if x.size(0) > 100:
            delta_ops = delta_ops * x.size(0) / 2
        #delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'Upsample', 'SliceMul', 'BatchNorm1d', 'LayerNorm']:
        delta_params = get_layer_param(layer)
        delta_ops = 0.0

    elif type_name in ['Upsample', 'MMUpsample', 'H_SigmoidQuant', 'Hardtanh', 'QuantizedHardtanh', 'AdaptiveMaxPool2d', 'MaxPool2d', 'Cat', 'AvgQuant', 'Reorg', 'SigmoidQuant', 'Softmax', 'Sigmoid', 'SigmoidOp', 'HardQuant', 'Sequential', 'WeightCenter', 'UpsampleQuant']:
        delta_params = 0
        delta_ops = 0.0
    ### unknown layer type
    
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, data):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    # data = Variable(torch.zeros(1, C, H, W))
    # data = data.cuda()
    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params

def count_meshnet(model, size, writer=None):

    latent_size = (1000, 512, 256, 128, 64)

    flops, params = measure_model(model, torch.zeros(2, 3, size, size))
    b_flops, b_params = measure_model(model.backbone, torch.zeros(2, 3, size, size))
    d_flops, d_params = flops - b_flops, params - b_params
    try:
        writer.print_str('backbone FLOPS: ' + str(b_flops/1e6) + ' backbone Param:' + str(b_params/1e6))
        writer.print_str('spiral decoder FLOPS: ' + str(d_flops/1e6) + ' spiral decoder Param:' + str(d_params/1e6))
        writer.print_str('total FLOPS: ' + str(flops/1e6) + ' total Param:' + str(params/1e6))
    except:
        pass
    print('backbone FLOPS: ' + str(b_flops/1e6) + ' backbone Param:' + str(b_params/1e6))
    print('spiral decoder FLOPS: ' + str(d_flops/1e6) + ' spiral decoder Param:' + str(d_params/1e6))
    print('total FLOPS: ' + str(flops/1e6) + ' total Param:' + str(params/1e6))

    # flops, params = measure_model(model.decoder_prior, [torch.zeros(1, latent_size[1], size//32, size//32), torch.zeros(1, latent_size[2], size//16, size//16),
    #                                                     torch.zeros(1, latent_size[3], size//8, size//8), torch.zeros(1, latent_size[4], size//4, size//4)])
    # writer.print_str('decoder_prior FLOPS: ' + str(flops / 1e6) + ' decoder_prior Param:' + str(params / 1e6))
    #
    # flops, params = measure_model(model.backbone_mt, torch.zeros(1, latent_size[4]+21, size//2, size//2))
    # writer.print_str('backbone_mt FLOPS: ' + str(flops/1e6) + ' backbone_mt Param:' + str(params/1e6))
    #
    # flops, params = measure_model(model.decoder2d, [torch.zeros(1, latent_size[1], size // 32, size // 32), torch.zeros(1, latent_size[2], size // 16, size // 16),
    #                                                 torch.zeros(1, latent_size[3], size // 8, size // 8), torch.zeros(1, latent_size[4], size // 4, size // 4)])
    # writer.print_str('decoder2d FLOPS: ' + str(flops / 1e6) + ' decoder2d Param:' + str(params / 1e6))
    #
    # flops, params = measure_model(model.decoder3d, torch.zeros(1, 256))
    # writer.print_str('decoder3d FLOPS: ' + str(flops / 1e6) + ' decoder3d Param:' + str(params / 1e6))


def count_multadds(model, size, writer=None):
    flops, params = measure_model(model, torch.zeros(2, 3, size[1], size[0]))
    try:
        writer.print_str('total FLOPS: ' + str(flops/1e6) + ' total Param:' + str(params/1e6))
    except:
        pass
    print('total FLOPS: ' + str(flops/1e6) + ' total Param:' + str(params/1e6))
