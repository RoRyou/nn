from __future__ import print_function
import torch
import numpy as np

# 创建一个Tensor并设置requires_grad=True:
x = torch.ones(2, 2, requires_grad=True)
print(x)
# grad_fn积分方法名，默认为None
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)
# 注意x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn

# 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None
# 叶子节点
print(x.is_leaf, y.is_leaf)

z = y * y * 3

out = z.mean()
print(z)
print(out)

# 通过.requires_grad_()来用in-place的方式改变requires_grad属性
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

#因为out是一个标量，所以调用backward()时不需要指定求导变量
out.backward()
print(x.grad)

# 再来反向传播一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_() # 清0
out3.backward()
print(x.grad)

x = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y = 2*x
z = y.view(2,2)
print(z)
#现在 z 不是一个标量，所以在调用backward时需要传入一个和z同形的权重向量进行加权求和得到一个标量。
v= torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
z.backward(v)
print(x.grad)
#x.grad是和x同形的张量。
x = torch.tensor(1.0,requires_grad=True)
y1 = x **2
with torch.no_grad():
    y2=x**3
y3 = y1+y2
print(x.requires_grad)
print(y1,y1.requires_grad)
print(y2,y2.requires_grad)
print(y3,y3.requires_grad)

y3.backward()
print(x.grad)

#如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我么可以对tensor.data进行操作
x = torch.ones(1,requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)

#线性回归输出是一个连续值，因此适用于回归问题。回归问题在实际中很常见，如预测房屋价格、气温、销售额等连续值的问题。与回归问题不同，分类问题中模型的最终输出是一个离散值。我们所说的图像分类、垃圾邮件识别、疾病检测等输出为离散值的问题都属于分类问题的范畴。softmax回归则适用于分类问题。

#由于线性回归和softmax回归都是单层神经网络，它们涉及的概念和技术同样适用于大多数的深度学习模型。我们首先以线性回归为例，介绍大多数深度学习模型的基本要素和表示方法



# 当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。
# 本节使用的线性回归和平方误差刚好属于这个范畴。
# 大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。


