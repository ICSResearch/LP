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
from torch.utils.data import DataLoader
import random


def get_network(args):
    """ return given network
    """

    if args.net == 'resnet50':
        from .models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet_LPA':
        from .models.resnet_LPA import resnet50
        net = resnet50()
    elif args.net == 'resnet_LPC':
        from .models.resnet_LPC import resnet50
        net = resnet50()
    elif args.net == 'vgg16':
        from .models.vgg import vgg16_bn
        net = vgg16_bn(pretrained=False, num_classes=365)
    elif args.net == 'vgg16_lpC':
        from .models.vgg_LPC import vgg16_bn
        net = vgg16_bn(pretrained=False, num_classes=365)
    elif args.net == 'vgg16_lpA':
        from .models.vgg_LPA import vgg16_bn
        net = vgg16_bn(pretrained=False, num_classes=365)


    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
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
        #transforms.ToPILImage(),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    place365_training = torchvision.datasets.Places365(root='./data', small=True, download=False, transform=transform_train)

    place365_training_loader = DataLoader(
        place365_training, shuffle=shuffle, pin_memory=True, num_workers=num_workers, batch_size=batch_size)

    return place365_training_loader

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
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    place365_test = torchvision.datasets.Places365(root='/media/npu/Data/nc/data', split="val", small=True, download=False, transform=transform_test)
    place365_test_loader = DataLoader(
        place365_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return place365_test_loader

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


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]
    mean = torch.mean(train.cuda(), dim=(0, 2, 3))
    std = torch.std(train.cuda(), dim=(0, 2, 3))
    return mean, std



if __name__ == '__main__':
    # set = torchvision.datasets.Places365(root='/media/npu/Data/nc/data', small=True, download=False, transform=transforms.ToTensor())
    set = torchvision.datasets.CelebA(root='/home/lagarto/edge_connect/dataset', download=True, transform=transforms.ToTensor())

    set = torch.utils.data.Subset(set, range(int(0.4 * len(set))))
    # set = torch.utils.data.Sampler(set)

    train_mean, train_std = get_mean_std(set)

    # test_mean, test_std = get_mean_std(test_dataset)

    print(train_mean, train_std)
    # print(test_mean, test_std)

    # print(set)
    res = get_mean_std(set)
    print(res)