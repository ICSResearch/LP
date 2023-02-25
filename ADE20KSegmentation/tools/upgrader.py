import argparse

import torch.nn as nn
import torch
import copy
from learning_padding import LearningPaddingAllStride,\
    LearningPaddingA_AllStride


class Upgrade:
    def __init__(self, net:nn.Module, type='LPC'):
        super(Upgrade, self).__init__()
        assert type in ['LPC', 'LPA', 'LPN']
        self.new_net = copy.deepcopy(net)
        if type == 'LPC':
            self.pad_block = LearningPaddingAllStride
        elif type == 'LPA':
            self.pad_block = LearningPaddingA_AllStride

        self.convert_pad(self.new_net)

    def freeze(self):
        for m in self.new_net.modules():
            for parm in m.parameters():
                parm.requires_grad = False
        self.unfreeze_pad(self.new_net)


    def unfreeze(self):
        for p in self.new_net.parameters():
            p.requires_grad = True

    def convert_pad(self, mod):
        for child_name, child in mod.named_children():
            if isinstance(child, nn.Conv2d) and child.padding > (0, 0):
                pad = nn.Sequential()
                # for pad_size in range(child.padding[0]):
                #     pad.add_module('lp' + str(pad_size), self.pad_block(child.in_channels))
                pad.add_module('lp', self.pad_block(child.in_channels, child.padding[0]))
                child.padding = (0, 0)
                pad.add_module('conv', child)
                setattr(mod, child_name, pad)
            else:
                self.convert_pad(child)

    def unfreeze_pad(self, mod):
        for child_name, child in mod.named_children():
            if isinstance(child, self.pad_block):
                for p in child.parameters():
                    p.requires_grad = True
            else:
                self.unfreeze_pad(child)


