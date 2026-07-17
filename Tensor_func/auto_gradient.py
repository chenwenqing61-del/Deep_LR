import torch
# x = torch.tensor([1.0])
# y = torch.tensor([[3.0]])

# b = torch.rand(1,1,requires_grad=True)
# w = torch.rand(1,1,requires_grad=True)
# z = w*x +b
# print(b,w)
# loss = torch.nn.MSELoss()
# loss_value = loss(z,y)
# loss_value.backward()
# print(b.grad,w.grad)

x = torch.ones(2,2,requires_grad=True)
y = x*x
u = y.detach()
z = u*x
z.sum().backward()
print(x.grad==u)

