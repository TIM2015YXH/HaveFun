import os
import math
import logging

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cv2
import numpy as np
#import torch.cat as Cat
import torch.utils.model_zoo as model_zoo
try:
    from .bifpn import BIFPN
    #from .new_module import conv_layer, AvgQuant, mobile_unit, LinearUnit, linear_layer,AvgPool
    #from .new_module import QuantizedHardtanh, Cat, Conv2dQuant, AvgQuant, AvgPoolQuant, UpsampleQuant, conv_layer
    from .new_module import Cat
except:
    from bifpn import BIFPN
    from new_module import Cat

BN_McOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, cut=True, relu=True,
               group=1):
    #sequence = [Conv2dQuant(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=group)]
    sequence = [nn.Conv2d(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:

        if cut:
            sequence.append(nn.Hardtanh(0., 4., inplace=False))
        else:
            sequence.append(nn.ReLU(inplace=False))#Swish())

#     print('computation:', channel_in, channel_out, ks, stride)
    return nn.Sequential(*sequence)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def conv(numIn, numOut, k, s=1, p=0, bias=False):
    layers = []
    #layers.append(Conv2dQuant(numIn, numOut, k, s, p, bias=bias))
    layers.append(nn.Conv2d(numIn, numOut, k, s, p, bias=bias))
    layers.append(nn.BatchNorm2d(numOut))
    # layers.append(nn.ReLU(True))
    #alayers.append(QuantizedHardtanh(0, 4))
    nn.Hardtanh(0., 4., inplace=False)
    return nn.Sequential(*layers)

def mnconv(numIn, numOut, k, s=1, p=0, bias=False):
    if k < 2:
        return conv(numIn, numOut, k, s, p, bias=bias)
    layers = []
    #layers.append(Conv2dQuant(numIn, numIn, k, s, p, groups=numIn, bias=bias))
    layers.append(nn.Conv2d(numIn, numIn, k, s, p, groups=numIn, bias=bias))
    layers.append(nn.BatchNorm2d(numIn))
    # layers.append(nn.ReLU(True))
    #layers.append(QuantizedHardtanh(0, 4))
    nn.Hardtanh(0., 4., inplace=False)
    layers.append(conv(numIn, numOut, 1, bias=bias))
    return nn.Sequential(*layers)

def convBlock(numIn, numOut):
    layers = []
    layers.append(mnconv(numIn,  numOut, 3, 2, 1))
    layers.append(mnconv(numOut, numOut, 3, 1, 1))
    layers.append(mnconv(numOut, numOut, 3, 1, 1))
    return nn.Sequential(*layers)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
#         print('unit of mobile net block')
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
        self.conv2d = nn.Conv2d(self.channel_in, self.channel_out, 3, stride=1, padding=1)
        #self.avg = AvgQuant()

    def forward(self, x):
        #out = self.conv2d(x)#nn.Conv2d(self.channel_in, self.channel_out, 3, stride=1)
        #return out
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            #:out = self.avg([out, x])
            out = (out + x) / 2
        if self.has_half_out:
            return half_out, out
        else:
            return out

class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.cat = Cat()
        self.conv1 = mobile_unit(channel_in, channel_in//4)
        self.conv2 = mobile_unit(channel_in*5//4, channel_in//4)
        self.conv3 = mobile_unit(channel_in*6//4, channel_in//4)
        self.conv4 = mobile_unit(channel_in*7//4, channel_in//4)
    
    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = self.cat((x, out1))
        out2 = self.conv2(comb1)
        comb2 = self.cat((comb1, out2))
        out3 = self.conv3(comb2)
        comb3 = self.cat((comb2, out3))
        out4 = self.conv4(comb3)
        comb4 = self.cat((comb3, out4))
        return comb4

class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.cat = Cat()
        self.conv1 = mobile_unit(channel_in, channel_in//2)
        self.conv2 = mobile_unit(channel_in*3//2, channel_in//2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = self.cat((x, out1))
        out2 = self.conv2(comb1)
        comb2 = self.cat((comb1, out2))
        return comb2

class DenseBlock3(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock3, self).__init__()
        self.channel_in = channel_in
        self.cat = Cat()
        self.conv1 = mobile_unit(channel_in, channel_in)
        self.conv2 = mobile_unit(channel_in*2, channel_in)
        self.conv3 = mobile_unit(channel_in*3, channel_in)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = self.cat((x, out1))
        out2 = self.conv2(comb1)
        comb2 = self.cat((comb1, out2))
        out3 = self.conv3(comb2)
        comb3 = self.cat((comb2, out3))
        return comb3

class DenseBlock4(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock4, self).__init__()
        self.channel_in = channel_in
        self.cat = Cat()
        self.conv1 = mobile_unit(channel_in, channel_in//4)
        self.conv2 = mobile_unit(channel_in*5//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = self.cat((x, out1))
        out2 = self.conv2(comb1)
        comb2 = self.cat((comb1, out2))
        return comb2

class DenseBlock2_noExpand(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_noExpand, self).__init__()
        self.channel_in = channel_in
        self.cat = Cat()
        self.conv1 = mobile_unit(channel_in, channel_in*3//4)
        self.conv2 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = self.cat((x, out1))
        out2 = self.conv2(comb1)
        comb2 = self.cat((out1, out2))
        return comb2

class DenseBlock2_ShrinkHalf(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_ShrinkHalf, self).__init__()
        self.channel_in = channel_in
        self.cat = Cat()
        self.conv1 = mobile_unit(channel_in, channel_in//4)
        self.conv2 = mobile_unit(channel_in*5//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = self.cat((x, out1))
        out2 = self.conv2(comb1)
        comb2 = self.cat((out1, out2))
        return comb2

class base_model(nn.Module):
    dump_patches = True
    def __init__(self):
        super(base_model,self).__init__()
        self.layer0 = nn.Sequential(
        conv_layer(3, 16, 3, 1, 1),
        DenseBlock2_noExpand(16),
        DenseBlock2_noExpand(16),
        #nn.AvgPool2d(2),
        DenseBlock2(16),
        #AvgPoolQuant(2),
        DenseBlock4(32),
        nn.AvgPool2d(2),
        DenseBlock2_noExpand(48),
        DenseBlock2_noExpand(48),
        DenseBlock2(48),
        #AvgPoolQuant(2),
        nn.AvgPool2d(2),
        #DenseBlock4(48),
        DenseBlock2(96),
        DenseBlock2_noExpand(192),
        nn.AvgPool2d(2)) #32*32

        self.layer1 = nn.Sequential(
        DenseBlock2_noExpand(192),
        DenseBlock2_noExpand(192),
        #AvgPoolQuant(2),
        nn.AvgPool2d(2),
        #nn.AvgPool2d(2),
        DenseBlock(192),
        #DenseBlock2_noExpand(96),
        DenseBlock2_noExpand(384)) #16*16

        self.layer2 = nn.Sequential(
        #AvgPoolQuant(2),
        nn.AvgPool2d(2),
        DenseBlock2_noExpand(384),
        DenseBlock2_noExpand(384)) #8*8

        self.layer3 = nn.Sequential(
        #AvgPoolQuant(2),
        nn.AvgPool2d(2),
        #DenseBlock2_noExpand(96),
        conv(384, 384, 1), #4*4
        conv(384, 384, 1)) #4*4

        self.layer4 = nn.Sequential(
        #AvgPoolQuant(2),
        nn.AvgPool2d(2),
        conv(384, 384, 1),
        conv(384, 384, 1)) #2*2
        
        #self.avg = AvgQuant()
        
        self.neck = BIFPN(in_channels=[192, 384, 384, 384, 384],
                                out_channels=88,
                                stack=3,
                                num_outs=5)
        
        self.upsample_layer5 = nn.Sequential(
        #UpsampleQuant(scale_factor=2),
        nn.Upsample(scale_factor=2),
        #mobile_unit(88, 88),
        #mobile_unit(88, 88),
        #amobile_unit(44, 22),
        conv(88,176, 1), 
        conv(176, 88, 1))
    
        
        
    def forward(self,x):
        x0 = self.layer0(x)
  
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_fpn, _, _, _, _= self.neck([x0, x1, x2, x3, x4])
        us5 = self.upsample_layer5(x_fpn ) #64 + 192 * 64 * 24

        return us5
    
class PoseResNet(nn.Module):

    def __init__(self, heads, head_conv, base):
        super(PoseResNet, self).__init__()
        self.inplanes = 64
        self.heads = heads
        self.ich = 24 
        self.basenet = base()
        head_conv = 88

        # self.seg_head = nn.Sequential(
        #     mobile_unit(88, head_conv),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True)
        #     )
        

#         print("self.heads:\t", self.heads)
        for head in self.heads:
#             print("head:\t", head)
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  #nn.Conv2d(96, head_conv,
                  #  kernel_size=3, padding=1, bias=True),
                  mobile_unit(88, head_conv),
                  #nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(88, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
            
    def forward(self, x):
        x = self.basenet(x)

        ret = {}
        for head in self.heads:
            x1 = self.__getattr__(head)[0](x)
            ret[head] = self.__getattr__(head)[1](x1)
        # seg_out = self.seg_head(x)
        # ret['seg'] = seg_out

        return [ret]
    

def get_pose_net(num_layers, heads, head_conv=256, base = base_model):
  model = PoseResNet(heads, head_conv=head_conv, base = base)
  return model

def my_load_state_dict(curr_model, prev_model, start=0, end=None, to_cpu=True):
    own_state = curr_model.state_dict()
    prev_state = prev_model['state_dict']
    i = 0
    for name, param in list(prev_state.items())[start:end]:
        i += 1
        print(name)
        if name not in own_state:
            #name = name[6:]
            import ipdb; ipdb.set_trace()
        if name not in own_state:
            raise KeyError('Unexpected key "{}" in state_dict'.format(name))
        if isinstance(param, Parameter):
            param = param.data
        try:
            if to_cpu:
                own_state[name].copy_(param.cpu())
            else:
                own_state[name].copy_(param)
        except:
            print('Note: the parameter {} is inconsistent!'.format(name))
            continue

def load_model(pretrained):
    pretrained = torch.load(pretrained, map_location='cpu')
    import ipdb; ipdb.set_trace()
    #model = HandModel(input_channel=16, nclass=len(cfg['classes']), export_model=True)
    heads = {'hm': 1, 'wh': 2, 'reg': 2  }
    model = get_pose_net(0, heads)
    #model = torch.nn.DataParallel(model)
    my_load_state_dict(model, pretrained)
    model = model.cpu().eval()
    return model

def load_image(image_path):
    image = cv2.imread(image_path)
    h, w, c = image.shape
    maxw = max(h, w)
    t = int((maxw - h) / 2)
    l = int((maxw - w) / 2)
    expand_image = np.zeros((maxw, maxw, c), dtype=image.dtype)
    expand_image[t:t+h, l:l+w] = image
    image = cv2.resize(expand_image, dsize=(256, 256))
    yuv = image.copy().astype(np.float32)
    yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
    yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
    image = yuv
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image[image < 0] = 0
    image[image > 255] = 255
    image = np.round(image).astype(np.uint8).astype(np.float32) / 255.0 * 1

    image_tensor = torch.from_numpy(image).float().contiguous()
    image_tensor = image_tensor.view((1, 3, 256, 256)).cpu()
    return image_tensor

if __name__ == '__main__':
    from util import measure_model
    
    heads = {'hm': 1,
            'wh': 2 , 'reg':2 }
    model = get_pose_net(0, heads)
    input = torch.randn([1, 3, 256, 256])
    out = model(input)
    print(out[0]['seg'].shape)
    count_ops, count_params = measure_model(model, 256, 256)
    print('count_operations: ', count_ops/1e6, 'count_params: ', count_params / 1e6)
    
