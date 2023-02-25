#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
# from upgrader import Upgrade

if __name__ == '__main__':
    torch.cuda.set_device(0)
    # print(torch.__version__)
    # checkpoint/vgg16_lpC/Wednesday_28_April_2021_13h_04m_21s/vgg16_lpC-200-regular.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='seresnet152_as', help='net type')
    parser.add_argument('-weights', type=str,
                        default='checkpoint/seresnet152_as/Sunday_26_September_2021_17h_09m_05s/seresnet152_as-185-best.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    # net.load_state_dict(torch.load(args.weights), torch.device('cpu'))
    # dic = net.state_dict()
    # pretrain = {k: v for k, v in torch.load(args.weights).items() if k in dic.keys()}
    # net.load_state_dict(pretrain, torch.device('cuda:0'))
    # up = Upgrade(net)
    # net = up.new_net.cuda()
    # print(torch.load(args.weights, map_location=torch.device("cuda:3")).keys())
    net.load_state_dict(torch.load(args.weights, map_location=torch.device("cuda:0")))
    # net._save_to_state_dict('checkpoint/resnet152_lpC')
    # up = Upgrade(net)
    # net = up.new_net

    # torch.save(net.state_dict(), 'checkpoint/WRN40_lpA.pth')
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    # if args.gpu:
        # print('GPU INFO.....')
        # print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
