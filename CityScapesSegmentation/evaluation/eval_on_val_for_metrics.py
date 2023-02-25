# camera-ready

import sys
import os

sys.path.append("/root/deeplabv3")
from CityScapes.datasets import DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("/root/deeplabv3/model")
from CityScapes.model.deeplabv3 import DeepLabV3
# from CityScapes.model.deeplabv3_resnet50 import DeepLabV3

sys.path.append("/root/deeplabv3/utils")
from CityScapes.utils.utils import label_img_to_color

import torch
import json
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
# from upgraded import Upgrade
from upgrader import Upgrade

# torch.cuda.set_device(2)

trainId_to_id = {
    0: 7,
    1: 8,
    2: 11,
    3: 12,
    4: 13,
    5: 17,
    6: 19,
    7: 20,
    8: 21,
    9: 22,
    10: 23,
    11: 24,
    12: 25,
    13: 26,
    14: 27,
    15: 28,
    16: 31,
    17: 32,
    18: 33,
    19: 0
}
trainId_to_id_map_func = np.vectorize(trainId_to_id.get)

batch_size = 1

network = DeepLabV3("eval_val", project_dir=r"")
# upgrade = Upgrade(network.resnet, "LPA")
upgrade = Upgrade(network, "LPA")
# network.resnet = upgrade.new_net.cuda()
# network = upgrade.new_net.cuda()
network = upgrade.new_net


network.load_state_dict(torch.load(r"",
                                   map_location=torch.device("cpu")))


val_dataset = DatasetVal(cityscapes_data_path="",
                         cityscapes_meta_path="")


num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         # num_workers=1
                                         )

with open("", "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
batch_losses = []
for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        # imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        imgs = Variable(imgs) # (shape: (batch_size, 3, img_h, img_w))
        # label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)) # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        # loss = loss_fn(outputs, label_imgs)
        loss = 0
        # loss_value = loss.data.cpu().numpy()
        loss_value = 0
        batch_losses.append(loss_value)

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = F.upsample(outputs, size=(1024, 2048), mode="bilinear") # (shape: (batch_size, num_classes, 1024, 2048))

        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, 1024, 2048))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, 1024, 2048))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            pred_label_img = pred_label_imgs[i] # (shape: (1024, 2048))
            img_id = img_ids[i]

            # convert pred_label_img from trainId to id pixel values:
            pred_label_img = trainId_to_id_map_func(pred_label_img) # (shape: (1024, 2048))
            pred_label_img = pred_label_img.astype(np.uint8)

            cv2.imwrite(network.model_dir + "/" + img_id + "_pred_label_img.png", pred_label_img)



val_loss = np.mean(batch_losses)
print ("val loss: %g" % val_loss)
