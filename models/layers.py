import torch
from torch import nn
import copy
import numpy as np


class StatTrack(nn.Module):
    def __init__(self):
        super(StatTrack, self).__init__()
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        self.batch_mean = torch.mean(x, dim=(0, 2, 3))
        self.batch_var = torch.var(x, unbiased=False, dim=(0, 2, 3))
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        opt,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        last=False
    ):
        self.input_scale = 0.0
        self.input_size = 0
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        # st = StatTrack()

        layer = [conv]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                # layer = [st, bn]
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                # layer = [conv, st, bn]
                layer = [conv, bn]

        if opt.activetype is not "None" and not last:
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(conv)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        self.input_scale = self.kernel_size**2 * input.size(2) * input.size(3) / self.stride**2
        # self.input_scale = (input.clone().detach()**2).mean().item()
        self.input_size = len(input.clone().detach().flatten())
        return self.block.forward(input)


class FCBlock(nn.Module):
    def __init__(self, opt, in_channels, out_channels, bias=False):
        super(FCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels
        self.out_features = out_channels
        lin = nn.Linear(in_channels, out_channels, bias=bias)

        layer = [lin]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                layer = [lin, bn]

        if opt.activetype is not "None":
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(lin)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)


def FinalBlock(opt, in_channels, bias=False):
    out_channels = opt.num_classes
    opt = copy.deepcopy(opt)
    if not opt.preact:
        opt.activetype = "None"
    return nn.Linear(in_channels, out_channels)


def InitialBlock(opt, out_channels, kernel_size, stride=1, padding=0, bias=False):
    in_channels = opt.in_channels
    opt = copy.deepcopy(opt)
    return ConvBlock(
        opt=opt,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
