import time
from resnet import resnet50 as zero
from resnet_lpC import resnet50 as lpc
from resnet_lpA import resnet50 as lpa
from resnet_pad import resnet50 as pad
from resnet_part import resnet50 as part
import torch

n1 = zero().cuda()
n2 = lpc().cuda()
n3 = lpa().cuda()
n4 = pad().cuda()
n5 = pad('reflect').cuda()
n6 = pad('replicate').cuda()
n7 = part().cuda()
criterion = torch.nn.MSELoss()
optim1 = torch.optim.SGD(n1.parameters(), 0.1)
optim2 = torch.optim.SGD(n2.parameters(), 0.01)
optim3 = torch.optim.SGD(n3.parameters(), 0.01)
optim4 = torch.optim.SGD(n4.parameters(), 0.1)
optim5 = torch.optim.SGD(n5.parameters(), 0.1)
optim6 = torch.optim.SGD(n6.parameters(), 0.1)
optim7 = torch.optim.SGD(n7.parameters(), 0.1)

res = 22

# 6.8054962158203125
# 3.7788963317871094
# 6.036231517791748
# 2.2011637687683105
# 1.694962978363037
# 2.397608757019043
# 5.263419151306152

input = torch.rand([100, 3, res, res]).cuda()
gt = torch.rand([100, 100]).cuda()

# n1(input)
def run(model, op, t=0):
    for _ in range(t):
        o = model(input)
        criterion(o, gt).backward()
        op.step()
    t1 = time.time()
    o = model(input)
    criterion(o, gt).backward()
    op.step()
    t2 = time.time()
    # return 1 / (t2-t1)
    return (t2-t1) / 100 * 1000
print(run(n3, optim3, 2))
print(run(n1, optim1))
print(run(n3, optim3, 1))
print(run(n2, optim2, 2))
print(run(n3, optim3, ))
print(run(n4, optim4))
print(run(n5, optim5))
print(run(n6, optim6))
print(run(n7, optim7))
# print(run(n7, optim7))