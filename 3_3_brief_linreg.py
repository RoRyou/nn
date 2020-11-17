import torch
from time import time
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
from torch.nn import init
import torch.nn as nn
# 3.3.1 生成数据集
# 我们生成与上一节中相同的数据集。其中features是训练数据特征，labels是标签
num_inputs = 2 #x1，x2 有几个x
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 3.3.2 读取数据
# PyTorch提供了data包来读取数据。由于data常用作变量名，我们将导入的data模块用Data代替。
# 在每一次迭代中，我们将随机读取包含10个数据样本的小批量。

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# 这里data_iter的使用跟上一节中的一样。让我们读取并打印第一个小批量数据样本。

for X, y in data_iter:
    print(X, y)
    break


#
# 3.3.3 定义模型
# 在上一节从零开始的实现中，我们需要定义模型参数，并使用它们一步步描述模型是怎样计算的。
# 当模型结构变得更复杂时，这些步骤将变得更繁琐。
# 其实，PyTorch提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。
# 下面将介绍如何使用PyTorch更简洁地定义线性回归。
#
# 首先，导入torch.nn模块。
# 实际上，“nn”是neural networks（神经网络）的缩写。
# 顾名思义，该模块定义了大量神经网络的层。
# 之前我们已经用过了autograd，而nn就是利用autograd来定义模型。
# nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
# 在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。
# 下面先来看看如何用nn.Module实现一个线性回归模型。

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)  # 使用print可以打印出网络的结构

# 事实上我们还可以用nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中。

# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)#in_features = 2, out_features = 1 输入为2维，输出为1维
    # 此处还可以传入其他层
)

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict

net = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
    # ......
]))

print(net)
print(net[0])
#
# 可以通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个生成器。

for param in net.parameters():
    print(param)

# 回顾图3.1中线性回归在神经网络图中的表示。
# 作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。
# 因此，线性回归的输出层又叫全连接层。
# 注意：torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用input.unsqueeze(0)来添加一维。

# 3.3.4 初始化模型参数
# 在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。
# PyTorch在init模块中提供了多种参数初始化方法。
# 这里的init是initializer的缩写形式。
# 我们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。


init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
# 注：如果这里的net是用3.3.3节一开始的代码自定义的，那么上面代码会报错，net[0].weight应改为net.linear.weight，bias亦然。因为net[0]这样根据下标访问子模块的写法只有当net是个ModuleList或者Sequential实例时才可以，详见4.1节。

# 3.3.5 定义损失函数
# PyTorch在nn模块中提供了各种损失函数，这些损失函数可看作是一种特殊的层，PyTorch也将这些损失函数实现为nn.Module的子类。
# 我们现在使用它提供的均方误差损失作为模型的损失函数。

loss = nn.MSELoss()

# 3.3.6 定义优化算法
# 同样，我们也无须自己实现小批量随机梯度下降算法。
# torch.optim模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。
# 下面我们创建一个用于优化net所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法。

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 我们还可以为不同子网络设置不同的学习率，这在finetune时经常用到。例：
#
# optimizer = optim.SGD([
#     # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#     {'params': net.subnet1.parameters()},  # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}], lr=0.03)
# 有时候我们不想让学习率固定成一个常数，那如何调整学习率呢？
# 主要有两种做法。一种是修改optimizer.param_groups中对应的学习率，
# 另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。
# 但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。

# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率为之前的0.1倍

# 3.3.7 训练模型
# 在使用Gluon训练模型时，我们通过调用optim实例的step函数来迭代模型参数。
# 按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从而对批量中样本梯度求平均。

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 下面我们分别比较学到的模型参数和真实的模型参数。
# 我们从net获得需要的层，并访问其权重（weight）和偏差（bias）。学到的参数和真实的参数很接近。

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
