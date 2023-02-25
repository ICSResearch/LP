import argparse

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from torchstat import stat
import torchvision.models as models
from thop import profile
import torch.nn as nn
import copy
from learnning_padding import LearningPaddingC
import torch.nn.functional as F
# from upgraded import Upgrade as up_lpc
from upgrader import Upgrade
from fvcore.nn.flop_count import flop_count
from CityScapes.model.deeplabv3 import DeepLabV3
from place365.models.resnet import resnet50
from place365.models.vgg import vgg16_bn
from place365.models.vgg_LPC import vgg16_bn as lpc
from place365.models.vgg_LPA import vgg16_bn as lpa
# from place365.models.resnet_LP import resnet50 as lpc
# from place365.models.resnet_LPC import resnet50 as lpa

with torch.cuda.device(0):
    # n1 = DeepLabV3("tmp", "/home/lagarto/ lp/CityScapes/pretrained_models").eval()
    # n2 = DeepLabV3("tmp", "/home/lagarto/ lp/CityScapes/pretrained_models").eval()
    # n2.resnet = Upgrade(n2.resnet).new_net
    # n3 = DeepLabV3("tmp", "/home/lagarto/ lp/CityScapes/pretrained_models").eval()
    # n3 = Upgrade(n3).new_net
    n1 = vgg16_bn()
    n2 = lpc()
    n3 = lpa()


    # input = torch.randn(1, 3, 256, 256).cuda()
    input = torch.randn(1, 3, 256, 256)
    gflop_dict1, _ = flop_count(n1, input)
    gflops1 = sum(gflop_dict1.values())
    gflop_dict2, _ = flop_count(n2, input)
    gflops2 = sum(gflop_dict2.values())
    gflop_dict3, _ = flop_count(n3, input)
    gflops3 = sum(gflop_dict3.values())


    print("Ori: {}".format(gflops1))
    print("LPC: {}".format(gflops2))
    print("LPA: {}".format(gflops3))
