""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
from .AS import ASLinear, SqueezeAS, MaskLinear, StdGate, AvgPool, StdMaskLinear, ASFC,\
    DownSamplePool, DpMaskLinear, UpSamplePool, StdPool
import time
import math


class ASMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = ASLinear(in_features, hidden_features, hidden_features//64, drop)
        self.act = act_layer()
        # self.fc2 = SqueezeAS(hidden_features//4, hidden_features//64, drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.fc2 = MaskLinear(hidden_features, out_features, hidden_features//64, drop, False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ASMlp2(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ASLinear(in_features, hidden_features//4, hidden_features//32, drop)
        # self.act = act_layer()
        self.fc2 = SqueezeAS(hidden_features//4, hidden_features//64, drop)
        self.fc3 = nn.Linear(hidden_features//4, out_features)
        # self.fc2 = MaskLinear(hidden_features, out_features, hidden_features//64, drop, False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x


class ASMlp3(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ASLinear(in_features, hidden_features//2, in_features//32, drop)
        # self.act = act_layer()
        self.fc2 = SqueezeAS(hidden_features//2, hidden_features//16, drop)
        self.fc3 = nn.Linear(hidden_features//2, out_features)
        # self.fc2 = MaskLinear(hidden_features, out_features, hidden_features//64, drop, False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x


class ASMlp4(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ASLinear(in_features, hidden_features//2, in_features//16, drop)
        # self.act = act_layer()
        # self.fc = SqueezeAS(in_features, in_features//16, drop)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        # self.fc2 = MaskLinear(hidden_features, out_features, hidden_features//64, drop, False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ASMlp_Final(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = hidden_features // 2

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = StdGate(hidden_features, hidden_features // 64, drop)

        self.drop = nn.Dropout(drop)
        self.avg = AvgPool(hidden_features//out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = self.avg(x)
        return x


class ASMlp_Best(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
        """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = hidden_features // 2


        # theta1 = self.get_theta(hidden_features//2)
        theta1 = self.get_theta(in_features)
        theta2 = self.get_theta(hidden_features)

        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.fc1 = DpMaskLinear(in_features, theta2, theta1)
        self.fc2 = StdGate(hidden_features, theta2, drop, bias=False, act=False, shortcut=True)

        self.drop = nn.Dropout(drop)
        # self.avg = AvgPool(hidden_features // out_features)
        self.avg = DownSamplePool(hidden_features // out_features)
        # self.fc1 = StdMaskLinear(in_features, hidden_features, theta1, act=False)
        # self.fc1_add = StdMaskLinear(hidden_features, hidden_features, theta2, act=False)
        # self.fc1_add = StdGate(hidden_features, hidden_features//in_features, drop, bias=False, act=False)
        # self.fc1_add = StdGate(hidden_features//2, theta1, drop, bias=False, act=False, shortcut=False)
        # self.fc1_add = StdGate(in_features, theta1, drop, bias=False, act=False, shortcut=False)
        # self.fc1_add = nn.Linear(theta2, hidden_features)
        # self.fc1_add = StdPool(kernel_size=3, stride=1, padding=1)
        # self.fc1 = DpMaskLinear(in_features, hidden_features, theta1, drop, act=False)
        # self.up = UpSamplePool(in_features, hidden_features//in_features//2)
        # self.up2 = UpSamplePool(in_features, hidden_features//in_features//2)
        # self.ds = DownSamplePool(4)
        # self.act = nn.ELU(alpha=5., inplace=True)
        self.act = nn.GELU()
        # self.act = nn.ReLU(inplace=True)
        # index = []
        # x = 0
        # y = hidden_features//2
        # for i in range(hidden_features//2):
        #     index.append(x)
        #     index.append(y)
        #     x += 1
        #     y += 1
        #
        # self.register_buffer('index', torch.LongTensor(index))
        # self.hidden_features = hidden_features

    # def init_weights(self):
    #     torch.nn.init.trunc_normal_(self.w)


    def get_theta(self, dim):
        # min = 1
        # max = 2
        # while not min <= int(dim ** 0.5) <= max:
        #     min *= 2
        #     max *= 2
        #
        # theta = min if max - int(dim ** 0.5) > int(dim ** 0.5) - min else max
        return 2**math.ceil(math.log(dim**0.5, 2))

    def forward(self, x):
        x = self.fc1(x)
        # x = self.drop(x)
        # add = self.fc1_add(x)
        # print(x.unsqueeze(-2).shape)
        # print(x.unsqueeze(-1).shape)
        # print(x.shape[:-3] + (-1, self.hidden_features))

        # x = torch.cat([(x**2).unsqueeze(-2), add.unsqueeze(-2)], dim=-2)
        # x = x.transpose(-1, -2).contiguous().view(x.shape[:-3] + (-1, self.hidden_features))
        # x = self.ds(x)
        # x = self.up2(x)
        # x = self.fc1_add(x)
        # x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = self.avg(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

import torch
class GatedMlp_AS(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # print(in_features)
        # self.fc1 = StdMaskLinear(in_features, hidden_features, in_features//32, drop, act=False)
        # self.act = nn.Identity()
        self.act = act_layer()
        # hidden_features //= 2
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            self.act = act_layer()
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
            self.act = nn.Identity()

        # self.fc2 = nn.Linear(hidden_features, out_features)
        # if hidden_features ** 0.5 == int(hidden_features ** 0.5):
        #     print('change')
        min = 1
        max = 2
        while not min <= int(hidden_features**0.5) <= max:
            min *= 2
            max *= 2

        theta = min if max-int(hidden_features**0.5) > int(hidden_features**0.5)-min else max


        self.fc2 = nn.Sequential(StdGate(hidden_features, theta, dropout=drop),
                                 AvgPool(hidden_features//out_features))


        # print(out_features)
        # self.fc2 = StdGate(hidden_features, out_features // 4, drop)
        # self.fc2 = StdMaskLinear(hidden_features, out_features, 16, drop, act=False)
        # self.fc2 = ASFC(hidden_features, out_features, 32, drop, act=False)
        # self.avg = AvgPool(hidden_features // out_features)
        # self.avg = nn.AvgPool1d(hidden_features // out_features, hidden_features // out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # t1 = time.time()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gate(x)
        x = self.fc2(x)

        # t2 = time.time()
        # c = self.avg2(x)
        # t3 = time.time()

        # x = self.avg(x)
        # t4 = time.time()
        # print('self{}1d{}'.format((t3 - t2)/(t4-t1), (t4-t3)/(t4-t1)))
        # print(torch.sum(c-x))

        return x


class GatedMlp_Best(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        theta1 = self.get_theta(in_features)
        theta2 = self.get_theta(hidden_features)

        self.fc1 = UpSamplePool(in_features, hidden_features)

        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            self.act = act_layer()
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
            self.act = nn.Identity()

        self.fc2 = StdGate(hidden_features, theta2, drop, bias=False, act=False)

        self.drop = nn.Dropout(drop)

        self.dp = DownSamplePool(hidden_features // out_features)

        self.drop = nn.Dropout(drop)

    def get_theta(self, dim):
        min = 1
        max = 2
        while not min <= int(dim ** 0.5) <= max:
            min *= 2
            max *= 2

        theta = min if max - int(dim ** 0.5) > int(dim ** 0.5) - min else max
        return theta

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.dp(x)

        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
