import torch
from torch import nn,optim

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Linear = nn.Linear(5,3)
        self.Linear.weight.data = torch.tensor([
            [0.1,0.2,0.3],
            [0.4,0.5,0.6],
            [0.7,0.8,0.9],
            [0.1,0.2,0.3],
            [0.4,0.5,0.6]
        ]).T
        self.Linear.bias.data = torch.tensor([1.0,2.0,3.0])
    def forward(self,x):
        return self.Linear(x)
    
model = Model()
input = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float)
target = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float)
output = model(input)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)
optimizer.zero_grad()
loss(output,target).backward()
optimizer.step()
for i in model.state_dict():
    print(i)
    print(model.state_dict()[i])