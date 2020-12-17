import torch
import numpy as np

# 创建一个没有初始化的张量（5*3的矩阵）
x = torch.empty(5, 3)
print(x)

# 创建一个随机初始化（符合均匀分布的初始化）的张量
x = torch.rand(5, 3)
print(x)

# 初始化一个全为0的张量，并指定数据类型
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 通过列表创建一个张量（是一个1维的）
x = torch.tensor([5.20, 10.24])
print(x)

# 创建一个三维的张量
x = torch.Tensor(2, 3, 4)
print(x)

# 查看张量的形状
print(x.size())


x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x)
print(y)
# 同维度可以直接加法运算
print(x + y)
print(torch.add(x, y))
# 给定一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 原位/原地操作(in-place）
y.add_(x)
print(y)

print(y[:, 1])

# 改变张量的形状
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 2)
print(x, x.size(), y, y.size(), z, z.size())

x = torch.rand(4, 4)  # 生成均匀分布的值
y = torch.randn(4, 4)  # 生成正态分布的值
print(x, x.size())
print(y, y.size())

# 仅包含一个元素的tensor，可以使用.item()来得到对应的python数值
x = torch.randn(1)
print(x, x.item())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)  # Torch张量和NumPy数组将共享它们的底层内存位置，因此当一个改变时,另外也会改变。


a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a, 1, out=a)
print(a)
print(b)

# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
