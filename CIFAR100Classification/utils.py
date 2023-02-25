""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import timm.data.auto_augment
from torch.utils.data import DataLoader
from autoaugment import CIFAR10Policy
from timm.data.transforms_factory import create_transform


def get_network(args):
    """ return given network
    """
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg16_part':
        from models.vgg_part import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg16_lpC':
        from models.vgg_lpC import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg16_lpA':
        from models.vgg_lpA import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg16_lpA':
        from models.vgg_lpA import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'densenet201_part':
        from models.densenet_part import densenet201
        net = densenet201()
    elif args.net == 'densenet201_lpC':
        from models.densenet_lpC import densenet201
        net = densenet201()
    elif args.net == 'densenet201_lpA':
        from models.densenet_lpA import densenet201
        net = densenet201()
    elif args.net == 'densenet201_lpC':
        from models.densenet_lpC import densenet201
        net = densenet201()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'vgg16_circular':
        from models.vgg_pad import vgg16_bn
        net = vgg16_bn('circular')
    elif args.net == 'vgg16_reflect':
        from models.vgg_pad import vgg16_bn
        net = vgg16_bn('reflect')
    elif args.net == 'resnet50_circular':
        from models.resnet_pad import resnet50
        net = resnet50(pm='circular')
    elif args.net == 'resnet101_circular':
        from models.resnet_pad import resnet101
        net = resnet101(pm='circular')
    elif args.net == 'resnet50_reflect':
        from models.resnet_pad import resnet50
        net = resnet50(pm='reflect')
    elif args.net == 'resnet101_reflect':
        from models.resnet_pad import resnet101
        net = resnet101(pm='reflect')
    elif args.net == 'resnet50_replicate':
        from models.resnet_pad import resnet50
        net = resnet50(pm='replicate')
    elif args.net == 'resnet101_replicate':
        from models.resnet_pad import resnet101
        net = resnet101(pm='replicate')
    elif args.net == 'resnet50_zero':
        from models.resnet_pad import resnet50
        net = resnet50(pm='zeros')
    elif args.net == 'resnet18_zero':
        from models.resnet_pad import resnet18
        net = resnet18(pm='zeros')
    elif args.net == 'resnet50_part':
        from models.resnet_part import resnet50
        net = resnet50()
    elif args.net == 'resnet50_lpC':
        from models.resnet_lpC import resnet50
        net = resnet50()
    elif args.net == 'resnet50_lpA':
        from models.resnet_lpA import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet101_part':
        from models.resnet_part import resnet101
        net = resnet101()
    elif args.net == 'resnet101_lpC':
        from models.resnet_lpC import resnet101
        net = resnet101()
    elif args.net == 'resnet101_lpA':
        from models.resnet_lpA import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'resnet152_part':
        from models.resnet_part import resnet152
        net = resnet152()
    elif args.net == 'resnet152_lpC':
        from models.resnet_lpC import resnet152
        net = resnet152()
    elif args.net == 'resnet152_lpA':
        from models.resnet_lpA import resnet152
        net = resnet152()
    elif args.net == 'vgg16_lpC':
        from models.vgg_lpC import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'shake_py':
        from models.PyramidNet import pyramidnet272
        net = pyramidnet272()
    elif args.net == 'shake_py_lpC':
        from models.PyramidNet_lpC import pyramidnet272
        net = pyramidnet272()
    elif args.net == 'shake_py_lpA':
        from models.PyramidNet_lpA import pyramidnet272
        net = pyramidnet272()
    elif args.net == 'shake_py_part':
        from models.PyramidNet_part import pyramidnet272
        net = pyramidnet272()
    elif args.net == 'efficient_b0':
        from models.efficient import get_efficientnet
        net = get_efficientnet("efficientnetb0")
    elif args.net == 'efficient_b0_lpA':
        from models.efficient_lpA import get_efficientnet
        net = get_efficientnet("efficientnetb0", dataset='cifar100')
    elif args.net == 'efficient_b0_lpC':
        from models.efficient_lpC import get_efficientnet
        net = get_efficientnet("efficientnetb0", dataset='cifar100')
    elif args.net == 'efficient_b0_part':
        from models.efficient_part import get_efficientnet
        net = get_efficientnet("efficientnetb0", dataset='cifar100')
    elif args.net == 'WRN40':
        from models.WRN import wide_resnet40_10
        net = wide_resnet40_10(pretrained=False)
    elif args.net == 'WRN40_part':
        from models.WRN_part import wide_resnet40_10
        net = wide_resnet40_10(pretrained=False)
    elif args.net == 'WRN40_lpA':
        from models.WRN_lpA import wide_resnet40_10
        net = wide_resnet40_10(pretrained=False)
    elif args.net == 'WRN40_lpC':
        from models.WRN_lpC import wide_resnet40_10
        net = wide_resnet40_10(pretrained=False)
    elif args.net == 'WRN28':
        from models.WRN import wide_resnet28_10
        net = wide_resnet28_10(pretrained=False)
    elif args.net == 'WRN28_part':
        from models.WRN_part import wide_resnet28_10
        net = wide_resnet28_10(pretrained=False)
    elif args.net == 'WRN28_lpA':
        from models.WRN_lpA import wide_resnet28_10
        net = wide_resnet28_10(pretrained=False)
    elif args.net == 'WRN28_lpC':
        from models.WRN_lpC import wide_resnet28_10
        net = wide_resnet28_10(pretrained=False)
    elif args.net == 'ResNeXt29':
        from models.ResNexst import resnext29_8x64d
        net = resnext29_8x64d(pretrained=False, dataset='cifar100')
    elif args.net == 'ResNeXt29_part':
        from models.ResNexst_part import resnext29_8x64d
        net = resnext29_8x64d(pretrained=False, dataset='cifar100')
    elif args.net == 'ResNeXt29_lpA':
        from models.ResNexst_lpA import resnext29_8x64d
        net = resnext29_8x64d(pretrained=False, dataset='cifar100')
    elif args.net == 'ResNeXt29_lpC':
        from models.ResNexst_lpC import resnext29_8x64d
        net = resnext29_8x64d(pretrained=False, dataset='cifar100')
    elif args.net == 'reg400':
        from models.regnet import regnety_004
        net = regnety_004(num_classes=100)
    elif args.net == 'reg400_lpA':
        from models.regnet import regnety_004
        net = regnety_004(pt='LPA', num_classes=100)
    elif args.net == 'reg400_lpC':
        from models.regnet import regnety_004
        net = regnety_004(pt='LPC', num_classes=100)
    elif args.net == 'reg400_part':
        from models.regnet import regnety_004
        net = regnety_004(pt='part', num_classes=100)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        # net = torch.nn.DataParallel(net)
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # The tansform for regnet and efficientnet
    # transform_train = create_transform((3, 32, 32),
    #                                    True,
    #                                    auto_augment='original',
    #                                    mean=mean,
    #                                    std=std,
    #                                    vflip=0.2,
    #                                    re_prob=0.25,
    #                                    ratio=[3./4., 4./3.],
    #                                    scale=[0.08, 1.0],
    #                                    re_mode='pixel',
    #                                    interpolation=4,
    #                                    crop_pct=.7
    #                                    )

    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, pin_memory=False, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        # transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)

    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # for f in weight_files:
    #     print(f)

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    best_files = [w for w in files if w != '.ipynb_checkpoints' and re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]