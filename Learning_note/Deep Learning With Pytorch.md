#### 1激活函数

在Pytorch中已经包装好了各种激活函数，使用时**直接调用即可**

Sigmoid函数，Tanh函数，ReLU函数，Softmax函数


#### 2 参数初始化与正则化

##### 2.1 全连接层(nn.Linear)以及其的参数初始化

###### 2.1.1 全连接层
在神经网络中，**参数主要位于全连接层(Affine映射)**
Pytorch里面的**torch.nn**模块专门用于**神经网络的构建与训练**，其中有两个属性weight和bias

###### 2.1.2 参数初始化

**常数初始化：** 所有权重参数初始化为一个常数
```
import torch.nn as nn
linear = nn.Linear(5,2)
nn.init.zeros_(linear.weight)
print(linear.weight)
nn.init.ones(linear.weight)
print(linear.weight)
nn.init.constant_(linear.weight)
print(linear.weight)
```
- 权重不能初始化为0，这样就会导致该层的神经元全部失活
- 权重一般不能初始化为同一常数，这样反向传播权重全部都进行相同的更新
每一个神经元对应的权重与每一条数据的**全部属性**共同计算得到一个值的
`z=w_1*x_1+w_2*x_2+w_3*x_3...`

**秩初始化：** 将权重矩阵初始化为一个单位矩阵
```
import torch.nn as nn
linear = nn.Linear(5,2)
nn.init.eye_(linear.weight)
print(linear.weight)
```

**正态分布初始化：** 将权重矩阵初始化为按照均值和标准差的正态分布
```
import torch.nn as nn
linear = nn.Linear(5,2)
nn.init.normal_(linear.weight,mean=0.0,std=1.0)
print(linear.weight)
```

**Xaiver 初始化(Glorot初始化)**
Xaiver根据输入输出的神经元数量调整权重的初始范围，使得输入和输出方差相近。
- **适用于Sigmoid和Tanh激活函数**，能缓解梯度消失与爆炸问题
Xaiver正态分布初始化：mean=0，std=$\sqrt{2/(n_{in}+n_{out})}$
Xaiver均匀分布初始化：区间$[-\sqrt{6/(n_{in}+n_{out})},\sqrt{6/(n_{in}+n_{out})}]$

```
import torch.nn as nn
linear = nn.Linear(5,2)
nn.init.xaiver_normal_(linear.weight)
print(linear.weight)
nn.init.xavier_uniform_(linear.weight)
print(linear.weight)
```

**He 初始化 (Kaiming初始化)**
只根据**输入**的神经元数量调整权重的初始范围
- **适用于ReLU及其变体(如Leaky ReLU)**
He正态分布初始化：mean=0，std=$\sqrt{2/n_{in}}$
He均匀分布初始化：$[-\sqrt{6/n_{in}},\sqrt{6/n_{in}}]$

```
import torch.nn as nn
linear=nn.Linear(5,2)
nn.init.kaiming_normal_(linear.weight)
print(linear.weight)
nn.init.kaiming_uniform_(linear.weight)
print(linear.weight)
```

###### Dropout 随机失活

按照一定的概率在学习过程中随机关闭一些神经元，**防止过拟合**
```
import torch
import torch.nn as nn
dropout = nn.Dropout(p=0.5)
x = torch.randint(1,10,(10,),dtyptorch.float32)
print("Dropout前：",x)
print("Dropout后：",dropout(x))
```

##### 2.2 正则化


#### 3. 搭建神经网络

##### 3.1 自定义模型
在神经网络框架中，由**多个层**组成的组件称之为**模块**，在Pytorch中模型就算一个Module，各网络层、模块也是Module。
`Module是所有神经网络的基类`

定义一个Module，需要继承torch.nn.Module并实现两个方法：\_\_init\_\_()、forward()

```
import torch
import torch.nn as nn

class Model(nn.Module):
	def __init__(self):
		super(Module,self).__init__()
		self.linear1 = nn.Linear(3,4)
		nn.init.xavier_normal_(self.linear1.weight)
		self.linear2 = nn.Linear(4,4)
		nn.init.kaiming_normal_(self.linear2.weight)
		self.linear3 = nn.Linear(4,2)
		
	def forward(self,x):
		x = self.linear1(x)
		x = torch.Tanh(x)
		x = self.linear2(x)
		x = torch.ReLU(x)
		x = self.linear3(x)
		x = torch.softmax(x,dim=-1)
		return x

model = Model()
output = model(torch.randn(10,3))
print("输出：\n",output)
print()

print("模型参数：")
for name,param in model.named_parameters():
	print(name,param)
	print()
print("模型参数：\n",model.state_dict())
```
- 查看模型参数通过model.named_parameters() -> 返回(name,param)
- 直接得到全部参数 model.state_dict()

##### 3.2 查看模型结构和参数数量
**torchsummary库产查看模型结构与参数数量**
- torchsummary.summary
summary(model,input_size=(),batch_size=,decice='')

##### 3.3 使用Sequential构建模型

通过torch.nn.Sequential来构建模型，按照各层次序一次写入即可
```
model = nn.Sequential(
nn.Linear(3,4),
torch.Tanh(),
nn.Linear(4,4),
nn.ReLU(),
nn.Linear(4,2)
nn.Softmax(dim=1)
)
def init_weights(m):
	if m==nn.Linear:
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)

model.apply(init_weight)
output = model(torch.randn(10,3))
print("输出：\n",output)
```
当**模型较简单**就可以直接使用nn.Sequential来构建模型，不必自定义类就可以组合新的框架

#### 4 损失函数

##### 4.1 分类问题的损失函数
分类问题可分为**二分类**和**多分类**

###### 4.1.1 二分类问题
二分类问题使用**二元交叉熵损失函数**（Binary Cross-Entropy Loss）

```
import torch
import torch.nn as nn

target = torch.tensor([[1],[0],[0]])
input = torch.randn((3,1))
prediction = torch.sigmoid(input)
loss = nn.BCELoss()
loss_value = loss(prediction,target)
print(loss_value)
```

###### 4.1.2 多分类问题
多分类问题使用多累交叉熵损失函数(Categorical Cross-Entropy Loss)
在Pytorch中使用**torch.nn.CrossEntropyLoss()**
如果使用CrossEntropyLoss就不用再后面加上一个Softmax激活函数，它里面自带Softmax函数
```
import torch
import torch.nn as nn

target = torch.tensor([1,0,3,2,5,4])
input = torch.randn(6,8)
loss nn.CrossEntropyLoss()
print(loss(input,target))
```


##### 4.2 回归任务损失函数

###### 4.2.1 MAE
平均绝对误差(**MAE**),也称L1 Loss
$$L=\frac{1}{n}\sum_{i=1}^{n}{\left |{y_i-\hat{y_i}}\right|}$$

###### 4.2.2 MSE
均方误差(**MSE**),也称为L2 Loss
$$L=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2$$

###### 4.2.3 Smooth L1
平滑L1：
$$\operatorname{SmoothL1}=
\begin{cases}
\dfrac{1}{2}\left(y_i-\hat{y}_i\right)^2, & \left|y_i-\hat{y}_i\right|<1,\\[6pt]
\left|y_i-\hat{y}_i\right|-\dfrac{1}{2}, & \left|y_i-\hat{y}_i\right|\ge 1.
\end{cases}$$
当$|y_i-\hat{y_i}|<1$时使用L2 Loss，使得损失函数平滑可导；当误差大于1时，使用L1 Loss 降低异常值的影响

#### 5 参数更新方法

##### 5.1 Momentum(动量法)
**保留历史梯度并给予一定的权重，使其也参与到参数更新中**
![[Pasted image 20260721134056.png]]


##### 5.2 学习率衰减

###### 5.2.1 等间距衰减
**当训练进行到一定轮数时，学习率下降为此时学习率的指定倍数**
通过`torch.optim.lr_scheduler.StepLR(optimizer,step_size,gamma)`实现等间距衰减
- step_size 间隔
- optimizer 选择的优化器
- gamma 衰减倍率


###### 5.2.2 指定间距衰减
**当训练进行到指定的轮数，学习率下降特定倍数的策略**
通过`torch.optim.lr_sheduler.MultiStepLR(optimizer,milestones,gamma)`实现指定间距衰减
- optimizer 优化器
- milestones 指定衰减间隔
- gamma 指定衰减比例

###### 5.2.3 指数衰减
**学习率乘以gamma的轮数倍**
`torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma)`实现指数衰减
- optimizer 是学习率衰减优化器
- gamma 底数，$lr = lr* gamma^{epoch}$

##### 5.3 AdaGrad(自适应梯度)

AdaGrad会为每一个参数**适当地调整学习率**，并随着学习地进行，梯度随之逐渐减少
$$H=H+{grad}^2 $$ $$W=W-lr*grad/\sqrt{H} $$
以$y=0.05x_1^2+x_2^2$为例
```
def adagrad(X,lr,n_iters):
	X_arr = X.detach().numpy().copy()
	H = torch.zero_like(X_arr)
	for epoch in range(n_iters):
		grad = X_arr*2 * w.T
		grad.suqeeze_()
		H += grad**2
		X -= lr/(torch.sqrt(H)+1e-7) *grad
		X_arr = np.vstack([X_arr,X])
	return X_arr
```


##### 5.4 RMSProp(均方根传播)
在AdaGrad基础上进行的改进，并非将所有梯度一视同仁的相加，而是逐渐遗忘过去的梯度，采用指数移动加权平均，呈指数地减少过去梯度
$$H = \alpha H+(1-\alpha)grad^2$$
$$W = W-lr*grad/\sqrt{H}$$
- H为历史梯度平凡和的指数移动加权平均
- α为历史梯度的权重
通过`torch.optim.RMSProp()`并设置α权重使用RMSProp
以$y=0.05x_1^2+x_2^2$为例
```
def rmsprop(X,lr,alpha,n_iters):
	X_arr=X.detach().numpy().copy()
	H = torch.zeros_like(X)
	for epoch in range(n_iters):
		grad = 2*X * w.T
		grad.squeeze_()
		H = H*alpha+(1-alpha)*grad
		X = lr/(torch.sqrt(H)+1e-7) *grad
		X_arr = np.vstack([X_arr,X.detach().numpy()])
	return X_arr
```

##### 5.5 Adam(自适应据估计)
Adam 融合了Momentum和AdaGrad的方法
![[Pasted image 20260726222333.png]]
使用`torch.optim.adam(paramters,lr,betas=[],n_iters=)`
