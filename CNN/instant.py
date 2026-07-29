import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

def train(model,train_dataset,test_dataset,lr,epoch_num,batch_size,device):
    def init_weight(layer):
        if type(layer)==nn.Linear | type(layer)==nn.Conv2d:
            torch.init.xaiver_normal_(layer.weight)
    model.apply(init_weight)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epoch_num):
        model.train()
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        train_correct_accumulate=0
        loss_accumulate=0
        for batch_count,(X,y) in enumerate(train_loader):
            X,y =X.to(device),y.to(device)
            pred = model(X)
            loss_value=loss(pred,y)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            #累加损失
            loss_accumulate+=loss_value.item()
            #累加正确输出的数量
            _,pred = pred.max(1) #返回values,indices
            train_correct_accumulate+=pred.eq(y).sum()
            #打印进度条
            print(f"\repoch:{epoch:0>2}[{'='*(int((batch_count+1)/len(train_loader)*50)):<50}]",end="")
        this_loss = loss_accumulate/len(train_loader)
        this_train_correct = train_correct_accumulate/len(train_dataset)

        model.eval()
        test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        test_correct_accumulate=0
        with torch.no_grad():
            for X,y in test_loader:
                X,y=X.to(device),y.to(device)
                pred = model(X)
                _,pred = pred.max(1)
                test_correct_accumulate+=pred.eq(y).sum()
        this_test_correct = test_correct_accumulate/len(test_dataset)
        print(f" loss:{this_loss:.6f}, train_acc:{this_train_correct:.6f}, test_acc:{this_test_correct:.6f}")

fashion_mnist_train = pd.read_csv("data/fashion-mnist_train.csv")
fashion_mnist_test = pd.read_csv("data/fashion-mnist_test.csv")
#-1表示程序自动计算样本数量,1表示一个通道
X_train = torch.tensor(fashion_mnist_train.iloc[:,1:].values,dtype=torch.float32).reshape(-1,1,28,28)
y_train = torch.tensor(fashion_mnist_train.iloc[:,0].values,dtype=torch.int64)

X_test = torch.tensor(fashion_mnist_test.iloc[:,1:].values,dtype=torch.float32).reshape(-1,1,28,28)
y_test = torch.tensor(fashion_mnist_test.iloc[:,0].values,dtype=torch.int64)

# plt.imshow(X_train[12345,0,:,:],cmap="grey")
# plt.show()

#构建数据集
train_dataset=TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)

#搭建模型
model = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
    nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
    nn.Flatten(),
    nn.Linear(400,120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.Sigmoid(),
    nn.Linear(84,10),
)

# X = torch.rand(size=(1,1,28,28),dtype=torch.float32)
# for layer in model:
#     X=layer(X)
#     print(f"{layer.__class__.__name__:<12}output.shape:{X.shape}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model,train_dataset,test_dataset,lr=0.9,epoch_num=20,batch_size=256,device=device)
