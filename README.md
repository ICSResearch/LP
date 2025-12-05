# LP
The implementation of Learning-based Padding: From Connectivity on Data Borders to Data Padding.

111111111111
## Statement
- You can find LPC/LPA module in learning_padding.py

## Environment
- python 3.8
- pytorch 1.7.1 (>=1.2.0)

## Usage
### 1. Employing LP module for you network.
If you want to pad a C channels feature map M with S stride.
```
from learning_padding import LearningPaddingByConvolution, LearningPaddingByAttention


LPC_map = LearningPaddingByConvolution(in_channels=C, stride=S)
LPA_map = LearningPaddingByAttention(in_channels=C, stride=S)
```

### 2. If you want to employ LPC/LPA module for a pretrained network with zero padding.

1. import upgrade.py file.
```
from upgrade import Upgrade
```

2. Prepare your network and load the checkpoint.

3. Upgrade your network. You have to choose the padding type, 'LPC' or 'LPA'.
```
upgrade = Upgrade(net, 'LPA')
new_net = upgrade.new_net
```


## Pretrained weightes

The part of checkpoints of paper can be found in [Google Drive](https://drive.google.com/drive/folders/1wrzpuVms5Dfkfqh6Ziersmo7blNVx7Sf?usp=sharing). The all checkpoints of paper can be found in [Baidu Drive](https://pan.baidu.com/s/1aGO8ah-l7CG4wBWoOoQ4EA?pwd=pn6o).

## Cite

```
@article{NING2023106048,
title = {Learning-based padding: From connectivity on data borders to data padding},
journal = {Engineering Applications of Artificial Intelligence},
volume = {121},
pages = {106048},
year = {2023},
issn = {0952-1976},
author = {Chao Ning and Hongping Gan and Minghe Shen and Tao Zhang},
}
```



## Acknowledgement

This repo benefits from awesome works of [CIFAR100](https://github.com/weiaicunzai/pytorch-cifar100), [timm](https://github.com/rwightman/pytorch-image-models) [DeepLabV3](https://github.com/fregu856/deeplabv3),
[ConvNeXt](https://github.com/facebookresearch/ConvNeXt).


## Email

lagarto@mail.nwpu.edu.cn
