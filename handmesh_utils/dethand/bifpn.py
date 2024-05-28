import torch.nn as nn
import torch.nn.functional as F


#from .new_module import Conv2dQuant, AvgPoolQuant,  QuantizedHardtanh, UpsampleQuant, AddQuant, AvgQuant, conv_layer, AdaptiveAvgQuant
import torch
#from new_module import Conv2dQuant

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

def conv(numIn, numOut, k, s=1, p=0, bias=False):
    layers = []
  #a  layers.append(Conv2dQuant(numIn, numOut, k, s, p, bias=bias))
    layers.append(nn.Conv2d(numIn, numOut, k, s, p, bias=bias))
    layers.append(nn.BatchNorm2d(numOut))
    layers.append(nn.ReLU(inplace=False))
    #layers.append(QuantizedHardtanh(0, 4))

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
        #self.avg = nn.AvgPool2d
        #self.avg = AvgPoolQuant
        #aself.avg = AvgQuant()

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            #self.avg((out, x))
            out = (out + x ) / 2
        if self.has_half_out:
            return half_out, out
        else:
            return out



class BIFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,      #output layer number
                 start_level=0, 
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False, #
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()
        self.relu = nn.ReLU(inplace=False)
        for i in range(self.start_level, self.backbone_end_level):
            #l_conv = ConvModule(
            #    in_channels[i],
            #    out_channels,
            #    1,
            #    conv_cfg=conv_cfg,
            #    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #    activation=self.activation,
            #    inplace=False)
            #l_conv = nn.Conv2d(in_channels[i], out_channels, 1, 1, 0,bias=False)
            l_conv = conv(in_channels[i], out_channels, 1)
            #l_conv = mobile_unit(in_channels[i], out_channels)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels=self.backbone_end_level-self.start_level,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      activation=activation))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                #extra_fpn_conv = ConvModule(
                #    in_channels,
                #    out_channels,
                #    3,
                #    stride=2,
                #    padding=1,
                #    conv_cfg=conv_cfg,
                #    norm_cfg=norm_cfg,
                #    activation=self.activation,
                #    inplace=False)
                extra_fpn_conv = mobile_unit(in_channels[i], out_channels, stride=2)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](self.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps = 0.0001):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.eps = eps 
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU(inplace=False)
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU(inplace=False)
        self.upsample =  nn.Upsample(scale_factor=2, mode='nearest')
        #self.avg = AdaptiveAvgQuant()

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        for jj in range(2):
            for i in range(self.levels-1):  # 1,2,3
                #fpn_conv = nn.Sequential(
                #    ConvModule(
                #        channels,
                #        channels,
                #        3,
                #        padding=1,
                #        conv_cfg=conv_cfg,
                #        norm_cfg=norm_cfg,
                #        activation=self.activation,
                #        inplace=False)
                #        )
                fpn_conv = nn.Sequential(mobile_unit(channels, channels))
                self.bifpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        #w1 = self.relu1(self.w1)
        #w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        #w2 = self.relu2(self.w2)
        #w2 /= torch.sum(w2, dim=0) + self.eps # normalize 
        # build top-down
        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        #for in_tensor in inputs:
            #inputs_clone.append(in_tensor.clone())
       # for i in range(levels - 1, 0, -1):
       #     pathtd[i - 1] = self.avg((pathtd[i - 1], self.upsample(pathtd[i])))
       #     pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
       #     idx_bifpn = idx_bifpn + 1
#V

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = (pathtd[i - 1] +  self.upsample(pathtd[i])) / 2
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1

        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = (pathtd[i + 1] +  self.max_pool(pathtd[i]) +  inputs[i + 1]) / 3
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1

        pathtd[levels - 1] = (pathtd[levels - 1] + self.max_pool(pathtd[levels - 2])) / 2
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        
        #pathtd[levels - 1] = (pathtd[levels - 1] + F.max_pool2d(pathtd[levels - 2], kernel_size=2))
        #pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        #for i in range(levels - 1, 0, -1):
        #    pathtd[i - 1] = (pathtd[i - 1] + self.upsample(pathtd[i])) / 2
        #    pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
        #    idx_bifpn = idx_bifpn + 1
        # build down-top
        #for i in range(0, levels - 2, 1):
        #    pathtd[i + 1] = (pathtd[i + 1] + F.max_pool2d(pathtd[i], kernel_size=2) + inputs_clone[i + 1]) / 3 
        #    pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
        #    idx_bifpn = idx_bifpn + 1
        #pathtd[levels - 1] = (pathtd[levels - 1] + F.max_pool2d(pathtd[levels - 2], kernel_size=2)) / 2
        #pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        #pathtd[levels - 1] = (w1[0, levels-1] * pathtd[levels - 1] + w1[1, levels-1] * F.max_pool2d(pathtd[levels - 2], kernel_size=2))/(w1[0, levels-1] + w1[1, levels-1] + self.eps)
        #pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd
