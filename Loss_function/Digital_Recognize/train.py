import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNer
from model import TwoLayerNet
x_train,x_test,t_train,t_test = get_data()

network = TwoLayerNet(input_size = 784,hidden_size=50,output_size=10) #定义一个两层神经网络
iters_num=1000 # 定义迭代的次数
batch_size = 100# 每一个批次的大小
learning_rate = 0.1 #学习率

train_loss_list = [] #用于存放每一次训练的loss
train_acc_list = [] #用于存放 每一次训练的准确率
test_acc_list = [] #用于存放每一次测试的准确率

iter_per_epoch = max(train_size/batch_size,1) #计算每轮的迭代次数

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size) #从训练集里面随机选择batch_size个数据索引
    x_batch = x_train[batch_mask] #根据索引拿到训练数据
    t_batch = t_train[batch_mask] #根据索引拿到训练数据的标签

    grad = network.numetrical_gradient(x_batch,t_batch) #计算每一个参数的梯度
    for key in ("W1","b1","W2","b2"):
        network.params[key] -= learning_rate*grad[key] #更新每一个参数
    loss = network.loss(x_batch,t_batch) #计算当前的损失函数
    train_loss_list.append(loss) #记录当前的loss

    if i % iter_per_epoch == 0: #模型每训练一个epoch，就计算一次准确度
        train_acc = network.accuracy(x_train,t_train) #计算训练集的准确度
        test_acc = network.accuracy(x_test,t_test) #计算测试集的准确度
        train_acc_list.append(train_acc) #记录训练集的准确度
        test_acc_list.append(test_acc) #记录测试集的准确度 
        print("train acc, test acc |"+str(train_acc)+"," + str(test_acc)) #输出一次训练的准确度和测试的准确度

markers = {"train":"o","test":"s"} #
x =np.arange(len(train_acc_list))
plt.plot(x,train_acc_list,label="train_acc")
plt.plot(x,test_acc_list,label="test_acc",linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0,1.0)
plt.legend(loc='lower right')
plt.show()

