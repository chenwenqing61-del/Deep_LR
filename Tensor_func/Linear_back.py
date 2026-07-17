import torch
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.utils.data import TensorDataset,DataLoader

#准备数据阶段
X = torch.randn(100,1)
w = torch.tensor([2.5])
b = torch.tensor([5.2])
noise = torch.randn(100,1)
y = w*X+b+noise
dataset = TensorDataset(X,y)
dataloader = DataLoader(dataset,batch_size=10,shuffle=True)
#模型构建阶段
model = nn.Linear(in_features=1,out_features=1)
#loss与优化器选择
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=1e-3)

loss_list = []
for epoch in range(100):
    total_loss=0
    train_num=0
    for x_train,y_train in dataloader:
        y_pred = model(x_train)
        loss_value = loss(y_pred,y_train)
        loss_tem = loss_value.detach()  #防止计算图被记录进total_loss里面，我只需要他的值
        total_loss+=loss_tem  #这里应该写total_loss +=loss_value.item()
        train_num+=len(y_train)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    loss_list.append(total_loss/train_num)

print(model.weight,model.bias)
plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
