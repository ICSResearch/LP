# camera-ready

import sys

sys.path.append("/root/deeplabv3")
from CityScapes.datasets import DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("/root/deeplabv3/model")
# from CityScapes.model.deeplabv3 import DeepLabV3
from CityScapes.model.deeplabv3_resnet50 import DeepLabV3

sys.path.append("/root/deeplabv3/utils")
from CityScapes.utils.utils import label_img_to_color

import torch
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

batch_size = 1


network = DeepLabV3("eval_val", project_dir=r"")
upgrade = Upgrade(network, "LPA")
# network.resnet = upgrade.new_net.cuda()
# network = upgrade.new_net.cuda()
network = upgrade.new_net


val_dataset = DatasetVal(cityscapes_data_path="",
                         cityscapes_meta_path="")

network.load_state_dict(torch.load(r"",
                                   map_location=torch.device("cpu")))


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
    # if step > 3:
    #     break
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        # imgs = torch.Tensor(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        imgs = torch.Tensor(imgs) # (shape: (batch_size, 3, img_h, img_w))
        # label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)) # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        # loss = loss_fn(outputs, label_imgs)
        # loss_value = loss.data.cpu().numpy()
        loss_value = 0
        batch_losses.append(loss_value)

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            if i == 0:
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_id = img_ids[i]
                # img = imgs[i] # (shape: (3, img_h, img_w))

                # img = img.data.cpu().numpy()
                # img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                # img = img*np.array([0.229, 0.224, 0.225])
                # img = img + np.array([0.485, 0.456, 0.406])
                # img = img*255.0
                # img = img.astype(np.uint8)

                pred_label_img_color = label_img_to_color(pred_label_img)
                pred_label_img_color = pred_label_img_color[..., ::-1]


                # overlayed_img = 0.35*img + 0.65*pred_label_img_color
                # overlayed_img = overlayed_img.astype(np.uint8)
                # result = pred_label_img_color.astype(np.uint8)

                # cv2.imwrite(network.model_dir + "/" + img_id + "_overlayed.png", overlayed_img)
                cv2.imwrite(network.model_dir + "/" + img_id + "_result.png", pred_label_img_color)

val_loss = np.mean(batch_losses)
print ("val loss: %g" % val_loss)
