import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))
print(net)

X = torch.rand(3,4)
Y = net(X).sum()

print(Y)