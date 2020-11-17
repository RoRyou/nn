import torch
from time import time
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import random

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

start = time()
d = a + b
print(time() - start)

a = torch.ones(3)
b = 10
print(a + b)

# y = Xw + b + e
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 噪声项 ϵ 服从均值为0、标准差为0.01的正态分布
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# features的每一行是一个长度为2的向量，而labels的每一行是一个长度为1的向量（标量）
print(features[0], labels[0])


# 通过生成第二个特征features[:, 1]和标签 labels 的散点图，可以更直观地观察两者间的线性关系
def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)


# plt.show()


# 在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。
# 这里我们定义一个函数：它每次返回batch_size（批量大小）个随机样本的特征和标签。

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # shuffle() 方法将序列的所有元素随机排序。
    for i in range(0, num_examples, batch_size):
        #torch.LongTensor 是64位整型
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        # j是一个tensor
        yield features.index_select(0, j), labels.index_select(0, j)
        #




batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 3.2.3 初始化模型参数
# 我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

w = torch.tensor(np.random.normal(0, 0.1, (num_inputs, 1)), dtype=torch.float32)
b = torch.tensor(1, dtype=torch.float32)
# 之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True。
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 3.2.4 定义模型
# 下面是线性回归的矢量计算表达式的实现。我们使用mm函数做矩阵乘法
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b


# 3.2.5 定义损失函数
# 我们使用上一节描述的平方损失来定义线性回归的损失函数。
# 在实现中，我们需要把真实值y变形成预测值y_hat的形状。以下函数返回的结果也将和y_hat的形状相同。
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 3.2.6 定义优化算法
# 以下的sgd函数实现了上一节中介绍的小批量随机梯度下降算法。
# 它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。
# 我们将它除以批量大小来得到平均值
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# 3.2.7 训练模型
# 在训练中，我们将多次迭代模型参数。
# 在每次迭代中，我们根据当前读取的小批量数据样本（特征X和标签y），通过调用反向函数backward计算小批量随机梯度，并调用优化算法sgd迭代模型参数。
# 由于我们之前设批量大小batch_size为10，每个小批量的损失l的形状为(10, 1)。回忆一下自动求梯度一节。
# 由于变量l并不是一个标量，所以我们可以调用.sum()将其求和得到一个标量，再运行l.backward()得到该变量有关模型参数的梯度。
# 注意在每次更新完参数后不要忘了将参数的梯度清零。
# 在一个迭代周期（epoch）中，我们将完整遍历一遍data_iter函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。
# 这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设3和0.03。
# 在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。
# 而有关学习率对模型的影响，我们会在后面“优化算法”一章中详细介绍。

lr = 0.03
num_epochs = 3 #迭代周期个数
net = linreg #线性回归
loss = squared_loss #损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):# 在每次迭代中，我们根据当前读取的小批量数据样本（特征X和标签y）
        l = loss(net(X, w, b), y).sum() #损失函数 #由于变量l并不是一个标量，所以我们可以调用.sum()将其求和得到一个标量
        l.backward()#通过调用反向函数backward计算小批量随机梯度
        sgd([w, b], lr, batch_size)#并调用优化算法sgd迭代模型参数。

        w.grad.data.zero_ #清零
        b.grad.data.zero_
    train_l = loss(net(features, w, b), labels)
    print('epoch %d,loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
