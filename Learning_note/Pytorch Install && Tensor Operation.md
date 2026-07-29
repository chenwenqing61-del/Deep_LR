#### 1 Pytorch Install

`确定pytorch版本 -> 合适的cuda版本(需确定显卡允许的最高cuda版本) -> 合适的python版本` 
**官网直接查看版本关系**
需要创建虚拟环境用于管理python版本

#### 2 Tensor Operation

##### 2.1 张量的创建方法
###### 2.1.1 基本张量的创建

**创建指定内容的张量**
torch.tensor()进行创建

```
import torch
import numpy as np
#创建标量张量，默认dtype=float32
tensor1=torch.tensor(10)
print(tensor1)
#使用数值创建tensor
tensor2 = torch.tensor([10,9,8])
print(tensor2)
#使用 numpy 创建tensor
tensor3 = torch.tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
print(tensor3)
```

**创建指定形状的张量**
torch.Tensor(size)

```
import torch
#创建指定大小的tensor，默认为float32
tensor1 = torch.Tensor(3,2,4)
print(tensor1)
print(tensor1.dtype)

#可用于创建指定内容的tensor
tensor2 = torch.Tensor([2,3,4])
print(tensor2)
```

**创建指定类型的张量**
使用torch.IntTensor()、torch.LongTensor()、torch.ShortTensor()等进行创建指定类型的张量

也可以直接使用torch.tensor(dtype=datatype)进行数据类型指定，同时这里可以进行数据类型的强制转换

```
#创建int32类型的张量
tensor1 = torch.IntTensor(2,3)
print(tensor1)
#指定数据类型
tensor1 = torch.tensor([2,3],dtype=int32)
print(tensor1)
#强制进行数据类型转换
tensor1 = torch.tensor([2.5,3.1],dtype=int32) #将小数部分舍掉
```

###### 2.1.2指定区间张量的创建

**在区间按步长创建张量**
`torch.arange`(begin,end,step)

```
import torch
tensor1 = torch.arange(1,10,3)
print(tensor1)

tensor2 = torch.arange(6)
print(6)
```

**区间内按元素数量创建张量**
`torch.linspace`(begin,end,**steps**) steps指定生成的元素个数
```
import torch
#在指定区间里面生成9个元素
tensor1 = torch.linspace(10,30,9)
print(tensor1)
```

**在指数区间内按指定底数创建张量**
`torch.logspace`(begin,end,**steps**,**base**)，steps指定元素个数，base指定底数
```
import torch
#在1~10之间生成3个数，并将这个三个数作为2的指数
tensor1 = torch.logspace(1,10,3,2)
print(tensor1)
```

###### 2.1.3 按数值填充张量

`torch.zeros(size)` 创建指定形状的全为0的张量
`torch.ones(size)` 创建指定形状的全为1的张量
`torch.full(size,value)` 创建指定形状的全为value的张量
`torch.empty(size)` 创建指定形状的未初始化的张量
`torch.zeros_like(tensor_input)` 创建与给定张量形状相同的全0张量
`torch.ones_like(tensor_input)` 创建与给定张量形状相同的全1张量
`torch.full_like(tensor_input,value) `创建与给定张量形状相同的全为value的张量
`torch.empty_like(tensor_input)` 创建与给定张量形状相同的未初始化的张量

```
import torch
#创建一个3行4列的全为0矩阵
tensor1 = torch.zeros(3,4)
print(tensor1)
#创建一个与tensor1相同形状的全为1的矩阵
tensor2 = torch.ones_like(tensor1)
print(tensor2)
#创建一个与tensor1相同形状的全为5的矩阵
tensor3 = torch.full_like(tensor1,5)
print(tensor3)
#创建一个与tensor1形状相同的未初始化的矩阵
tensor4 = torch.empty_like(tensor1)
print(tensor4)
```

**创建单位矩阵**`torch.eye(n,m)`
```
import torch
#创建一个3×3的单位矩阵
tensor1 = torch.eye(3)
#创建一个3×4的单位矩阵(3×3的单位矩阵+一列全为0的矩阵)
tensor2 = torch.eye(3,4)
```

###### 2.1.4 随机张量创建

`torch.rand(size)`创建在\[0,1)均匀分布，指定形状的张量
`torch.randint(low,high,size)` 创建在\[low,high)均匀分布的，指定形状的张量
`torch.randn(size)` 创建标准正态分布的，指定形状的张量
`torch.normal(mean,std,size)` 创建自定义参数的正态分布的指定形状的张量
`torch.rand_like(tensor_input)`
`torch.randint_like(input,low,high)`
`torch.randn_like(tensor_input)`

```
import torch

tensor1 = torch.rand((3,4))
print(tensor1)
tensor2 = torch.randint_like(tensor1,1,5)
print(tensor2)
tensor3 = torch.normal(5,1,tensor1.shape) #tensor1.shape获取tensor1的形状
```

`torch.randperm(m)`生成从0~m-1的随机排列，洗牌

`torch.random.initial_seed()`查看随机种子
`torch.manual_seed(seed)`设置随机种子
```
import torch
print(torch.random.initial_seed())
torch.manual_seed(521)
print(torch.random.initial_seed())
```

##### 2.2 张量转换

###### 2.2.1 张量元素类似的转换

`Tensor.type(dtype)`**修改张量的类型**

```
import torch

tensor1 = torch.tensor([1,2,3])
print(tensor1,tensor1.dtype)
tensor1 = tensor1.type(torch.float32)
print(tensor1,tensor1.dtype)
```

`Tensor.double()`**等修改张量的类型**

double() / long() / short() ...直接修改张量数据类型

```
import torch
tensor1 = torch.tensor([1,2,3])
print(tensor1,tensor1.dtype)
tensor1.long()
print(tensor1,tensor1.dtype)
```

###### 2.2.2 Tensor与ndarray转换

共享内存的概念：就是两个变量里面的内容来自同一块存储，一个变化另一个就会跟着变化

**Tensor转为ndarray**
`Tensor.numpy()`方法，此时**共享内存**，使用.copy()方法不共享内存,`Tensor.numpy().copy()`不共享内存

```
import torch
tensor1 = torch.rand(3,2)
numpy_array = tensor1.nuumpy() #此时共享内存
print(tensor1)
print(numpy_array)
print(tensor1,numpy_array) #数据类型不同，一个是tensor一个是numpy
tensor1[:,0] = -1
#此时两者均发生变化
print(tensor1)
print(numpy_array)
numpy_array = tensor1.numpy().copy()
tensor1[:,0]=0
#由于使用了copy()方法，因此不共享内存，即为两个变量都单独开了一个内存
print(tensor1)
print(numpy_array)
```

**ndarray转tensor**
使用`torch.from_numpy(np)`，将np转为tensor，此时共享内存，加上`copy()`语句，可防止共享内存，即torch.from_numpy(np.copy())

```
import torch
import numpy as np
numpy_array = numpy.random.randn(3)
tensor1 = torch.from_numpy(numpy_array)

#此时由于是共享内存，因此两个变量的内容都会发生变化
numpy_array[0] = 3
print(numpy_array)
print(tensor1)

#使用copy()语句，为两个变量分别开放不同的内存
tensor1 = torch.from_numpy(numpy_array.copy())
numpy_array[0]=10
print(numpy_array)
print(tensor1)
```

**使用numpy创建tensor**
使用torch.tensor(np)，不会共享内存

###### 2.2.3 Tensor与标量的转换

对于**单元素**tensor，可以通过item()指令转换为标量
```
import torch
tensor1 = torch.tensor(1)
print(tensor1)
print(tensor1.item())
```

##### 2.3 张量数值运算

对于每一个运算，只有当在运算后面加上'\_'才会对张量本身数值进行改变
###### 2.3.1 基本四则运算
+、-、\*、/
不改变原张量值：add()，sub()，mul()，div()加减乘除运算
改变原张量值：add_()，sub_()，mul_()，div_()改变原来张量
通过**tensor**.进行操作运算

###### 2.3.2 取反运算

-、neg()、neg_()取反，在原数值加上一个负

###### 2.3.3 求幂、开根号、求对数、求指数

求幂：\*\*、pow()、pow_()指令
`tensor.pow(2) `=  `tensor**2` = `tensor.pow_(2)`

开根号：sqrt()、sqrt_()指令
`tensor.sqrt(3)` = `tensor.sqrt_(3)`

以e为底求对数：log()、log_()指令，括号里面什么都不用加
`tensor.log()` = `tensor.log_()`

以e为底求指数：exp()、exp_()指令，括号里面什么都不加
`tensor.exp()` = `tensor.exp_()`

###### 2.3.4 哈达玛积（元素级运算）

只对张量里面的元素一 一对应进行相乘，必须是**相同维度的矩阵**，
通过mul(tensor)和\*tensor进行对位乘法操作
当实现卷积运算时，使用的是哈达玛积（卷积核进行滑动）
```
import torch
tensor1 = torch.tensor([[1,2],[3,4],[5,6]])
tensor2 = tensor1 #这里会产生共享内存
print(tensor1.mul(tensor2))
```

###### 2.3.5 矩阵乘法

实现矩阵之间的乘法，即(3,2)×(2,3)=(3,3)的矩阵
`mm()`严格**二维矩阵**相乘
`matual()`和`@`支持**多维矩阵**相乘，是对**最后两个维度**进行矩阵乘法运算，其他维度相同；
注意：当两个矩阵的维度不对齐时(最后两个维度相同但是前面的维度我完全相同)，如果其中一个的
维度为1，广播后就可以进行运算了如：
```
(4,5,2)@(4,2,5)可以进行运算
(3,2,1)@(1,2)可以进行运算，可以进行广播为(1,1,2)
```
###### 2.3.6 内存节省

在进行一些操作时，Python会为计算结果新开一个内存，例如：`X=X@Y`会为`X@Y`开一个新的内存，再将X指向`X@Y`对应的地址
**必须要求计算出来的结果与X的尺寸要相同，否则报错**
```
import torch
X=torch.randint(1,9,(4,3,2))
Y=torch.randint(1,9,(4,2,3))
print(id(X))#id()是获取X在内存中的存储地址
X=X@Y
print(id(X))
#两次得到的存储地址不同，因此会产生新的内存地址

X[:]=X@Y
print(id(X))
#此时X的地址与前面的相同，即不会新开一个内存，节省内存空间
```

##### 2.4 张量运算函数

sum()求和
mean()求均值
std()求标准差
max()/min()求最大值和最小值以及他们的索引(需要**指定维度才会输出索引**)
argmax()/argmin()求最大值和最小值的索引
unique()去重
sort()排序

通过tensor.way()的方式进行调用函数

##### 2.5 张量索引操作

###### 2.5.1 简单索引
其与数组的\[]取索引相同
```
import torch

tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1)
print(tensor1[0])
```

###### 2.5.2 范围索引
与数组取索引相同不过多赘述

###### 2.5.3 列表索引
通过取索引\[]里面加上\[]来取得相关数据
```
tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1[[1,2],[2,3]]) 输出第0维第1和第二维第2 以及 第0维第2和第二维第3
print(tensor1[[[0],[1]],[1,2]]) 输出第0维第0和第一维1，2 和 第0维第1和第二维1 2
```

###### 2.5.4 布尔索引
通过判断索引对应的值是否满足相关条件
```
tensor1=torch.randint(1,9,(3,4,5))
mask = tensor1[:,1,:]>5
print(tensor1[mask]) #表示在mask为1的位置索引(dim0,dim2)作为tensor1的索引
tensor2 = tensor1.permute(1,2,0) 
print(tensor2)
```

##### 2.6 张量形状操作

###### 2.6.1 交换维度
**交换两个维度transpose()**
```
import torch
tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1)
print(tensor1.transpose(1,2)) #交换第一和第二维度
```

**重新排列多个维度permute()**
```
import

tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1)
print(tensor1.permute(2,0,1)) #(3,4,5)->(5,3,4)
```

###### 2.6.2 调整形状

**调整张量的形状reshape()**
```
tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1)
print(tensor1.reshape(5,12))
print(tensor1.reshape(3,-1)) #-1表示函数自己去寻址自适应的尺寸
```

**view()调整张量形状，需要内存连续。共享内存**
- is_contiguous()判断是否内存连续
- contiguous()转换为内存连续
```
import torch
tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1.is_contiguous())
print(tensor1.view(-1,10))
tensor1 = tensor1.view(-1,10)
tensor1 = tensor1.T
print(tensor1,is_contiguous())
tensor1 = tensor1.contiguous()
tensor1 = tensor1.view(-1)
```

###### 2.6.3 增加或删除维度

unsqueeze()为tensor增加一个维度，例如：(3,4,5) ->(1,3,4,5)
```
import torch
tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1)
print(tensor1.unsqueeze(dim=0))  #默认是dim=0
print(tensor1.unsequeeze(dim=-1)) #指在最后加一个维度(3,4,5,1)
```

sequeeze()删除大小为1的维度 例如：(3,4,5,1,1)->(3,4,5,1)

```
import torch
tensor1 = randint(1,9,(3,4,1,1))
print(tensor1)
print(tensor1.sequeenze_(),tensor1.shape)
```

###### 2.6.4张量拼接

torch.cat(\[],dim)按照已有维度进行拼接，除拼接维度以外的其他维度必须相同
```
import torch
tensor1 = torch.randint(1,9,(3,4,5))
tensor2 = torch.randint(1,9,(3,1,5))
print(tensor1,tensor2)
print(torch.cat([tensor1,tensor2],dim=1))
```

torch.stack(\[],dim)张量堆积，按照**新维度**进行堆积。所以张量的形状必须相同
- 指定那一个dim，就会在该dim进行堆积   stack(\[(3,4,5),(3,4,5)],dim=1) -> (3,2,4,5)
```
import torch
tensor1 = torch.randint(1,9,(3,4,5))
tensor2 = torch.randint(1,9,(3,4,5))
tensor3 = torch.stack([tensor1,tensor2],dim=2)
print(tensor3.shape)
```

##### 2.7 自动微分模块

训练神经网络时，框架根据设计好的模型构建一个**计算图**，用于跟踪哪些数据通过哪些组合产生输出，并使用**反向传播**算法根据**给定的损失函数**的梯度调整参数

![[Pasted image 20260716222158.png]]
其中，**叶子节点(最基础的节点x,w,b)**是不能进行in-place计算的，**即不能自修改**，且只有参数才会用到requires_grad=true (因为只有参数才能)  -- 默认requires_grad为True
**可以通过is_leaf进行判断参数是否为叶子节点** print(x.is_leaf)
```
import torch
x = torch.tensor([[1.0]])
y = torch.tensor([[3.0]])
w = torch.rand(1,1,requires_grad=true)
b = torch.rand(1,1,requires_grad=true)
z = w*x+b
loss = torch.nn.MSELoss()
loss_value = loss(z,y)
print("w的梯度为\",w.grad)
print("b的梯度为\",b.grad)
```

自动微分的关键是**记录节点的数据与运算**
- **数据**记录在张量的data属性里面，张量的数值以及**反向传播计算梯度所需要的中间量**
- **运算**记录在张量的grad_fn里面，主要记录的是**计算图这个过程**
`对于非叶子节点的梯度在反向传播时会被释放掉(除非设置参数retain_grad=True)`
**叶子节点的梯度会累计，需要使用optimizer.zero_grad()进行清空**

**当我们需要将某些计算结果移动到计算图以外，这时使用x.detach()返回一个新张量**
- 这个张量与原来张量的值相同，但是丢失计算图，即梯度不能在该新变量上面进行梯度下降

##### 2.8 机器学习步骤
使用Pytorch训练一个模型一般分为4个步骤
`准备数据->构建模型->定义损失函数和优化器->模型训练`
线性回归案例：
```
import torch
import matplotlib.pyplot as plt
from torch impot nn,optim
from torch.utils.data import Tensordataset,Dataloader

X = torch.randn(100,1)
w = torch.tensor([2.5])
b = torch.tensor([5.2])
noise = torch.randn(100,1)
y = w*X+b+noise
dataset = Tensordataset(X,y)
dataloader = DataLoader(dataset,batch_size=10,shuffle=True)

model = nn.Linear(in_feature=1,out_feature=1)
optimizer = optim.SGD(model.parameters(),lr=1e-3)
loss = nn.MSELoss()
loss_list = []

for epoch in range(1000)
	total_loss=0
	train_num=0
	for x_train,y_train in dataloader
		y_pre = model(x)
		loss_value = loss(y_pre,y_train)
		total_loss+=loss_value.item()
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
```

