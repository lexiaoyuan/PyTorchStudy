import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 2
out = z.mean()
print(z, out)

a = torch.ones(2, 2)
a = ((a * 3) / (a - 2))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.grad_fn)

out.backward()
print(x)
print(x.grad)

print(z.data)
print(z.data.norm())  # 默认求二范数

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# y.backward() # RuntimeError: grad can be implicitly created only for scalar outputs

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
