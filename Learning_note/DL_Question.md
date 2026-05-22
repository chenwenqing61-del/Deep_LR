#### 1 梯度消失与梯度爆炸

当反向传播进行很多层的时候，每一层都对前一层梯度乘以了一个系数。当这个系数比较小时(小于1)，越往前传递，梯度越小、训练越慢，就会导致梯度消失；当系数比较大时，则越往前传递就会越大，导致梯度爆炸。

#### 2 更新参数方法的优化 -- 使模型更快更准地到全局最小值

##### 2.1 SGD缺点

SGD：随机梯度下降，对权重的更新是$W=W-lr*W'$ 

实现代码：
```
def SGD():
	def __init__(self,lr=0.01):
		self.lr=lr
	def update(self,params,grads):
		for key in params.key():
			params[key]-=self.lr*grads[key]
```
缺点：
- 陷入局部最小值
- 会陷入马鞍点
- 收敛速度慢
- 学习率选择：学习率过大导致震荡或不收敛，过小则收敛速度慢


##### 2.2Momentum(动量)

改变原来的SGD更新方法，我们不仅考虑当前的梯度，我们还将之前所有的梯度变化都考虑进去。
更新方法：令v为之前所有的梯度变化(v为负值)
$$v=av-lr*W'$$$$W=W+v$$
代码实现：

```
class Momentum:
	def __init__(self,lr=0.01,momentum=0.9)：
		self.lr=lr
		self.momentum=momentum
		slef.v=None
	
	def update(self,params,grads):
		if self.v ==None:
			self.v={}
			for key,val in params.item():
				self.v[key]=np.zeros_like(val)
		for key in params.key():
			self.v[key] = self.v[key]*momentum-self.lr*params[key]
			params[key] +=self.v[key]
```
优点：
- 减缓优化过程中的震荡，收敛更加快
- 可以有效避免马鞍点

##### 2.3学习率衰减策略

在深度学习模型训练过程中调整最频繁的就是学习率，所以好的学习率可以使模型逐渐收敛并获得更高的精度。较大学习率，会使得模型刚开始收敛迅速，但是可能在最优解附近震荡不收敛；较小的学习率可以提高收敛精度，但训练速度慢。
因此提出**初期选择较大学习率，后面选择较小学习率**


**1）等间隔衰减** 

每间隔固定的epoch对lr进行**等比率衰减**
如：每隔7轮，lr变为变为原来的0.7倍

**2）指定间隔衰减**

指定在**特定的epoch**对lr进行定系数衰减。
例如：在20和100轮将lr变为0.7倍

**3）指数衰减**

学习率按照$f(x)=a^x,a<1$ 进行衰减。例如：使学习率以0.99为底数，epoch为指数的衰减

##### 2.4 AdaGrad(自适应梯度)

$$h=h+(W')^2$$
$$W-=lr*W'/\sqrt{h}$$
其中，h表示历史梯度平方和
学习越深入更新的幅度就越小，如果永无止境的学习更新量会变为0，完全不更新。
实现代码：
```
class AdaGrad:
	def __init__(self,lr=0.01):
		self.lr=lr
		self.h=None
	def updata(self,params,grads):
		if self.h==None:
			self.h={}
			for key,val in params.item():
				self.h[key]=np.zero_like(val)
		for key in params.key():
			h[key] += grads[key]*grads[key]
			params[key] -= self.lr*grads[key]/(np.sqrt(self.h[key]+1e-7)
```

##### 2.5 RMSProp

是在AdaGrad基础上改进的，并非将所有梯度都一视同仁，而是逐渐遗忘过去的梯度
$$h=ah+(1-a)W'^2$$
$$W-=lr*W'/\sqrt{h}$$
过去的梯度对目前的影响随指数级的衰减
代码实现：
```
class RMSProp:
	def __init__(self,lr=0.01,a=0.01):
		self.lr=lr
		self.a=a
		self.h=None
	def updata(self,params,grads):
		if self.h==None:
			for key,val in params.item():
				self.h[key]=np.zero_like(val)
		for key in params.key():
			self.h[key]=a*self.h[key]+(1-a)*grads[key]*grads[key]
			grads[key]-=self.lr*grads[key]/(np.sqrt(self.h[key]+1e-7)
```
##### 2.6 Adam

**Adam(自适应矩估计)**，融合了Momentum和RMSProp
$$v=a_1v+(1-a_1)W'$$
$$h=a_2h+(1-a_2)W'^2$$
进行**偏差修正** 对刚开始的v和h进行放大
$$\hat{v}=v/(1-a_1)$$
$$\hat{h}=h/(1-a_2)$$
$$W=W-lr*\hat{v}/\hat{h}$$
实现代码：
```
class Adam:
	def __init__(self,lr=0.01,a1=0.01,a2=0.01):
		self.lr = lr
		self.a1=a1
		self.a2=a2
		self.h=None
		self.v=None
	def update(self,params,grads):
		if self.h==None:
			for key,val in params.key():
				self.h = np.zero_like(val)
		if self.v==None:
			for key,val in params.item():
				self.v = np.zero_like(val)
		for key in params.key():
			self.v[key] = a1*self.v[key]+(1-a1)*grads[key]
			self.h[key] = a2*self.h[key]+(1-a2)*grads[key]*grads[key]
			self.v[key] = self.v[key]/(1-a1)
			self.h[key] = self.h[key]/(1-a2)
			grads[key] -=self.lr*self.v[key]/sqrt(self.h[key]+1e-7)
```

#### 3 参数初始化问题 -- 有效缓解梯度消失与梯度爆炸问题

在进行参数初始化时，一定不能使得整个W矩阵的值都相同，否则，在进行前向传播和反向传播的时候，W中的全部参数更新相同，即所有神经元都在干一件事 -- **权重均一化(对称性问题)**

- **秩初始化**，将权重矩阵初始化为单位矩阵 -- 主对角线元素为1，其余元素为0

- **正态分布初始化**，通过设定μ和σ对矩阵进行随机初始化

- **均匀分布初始化**，权重参数在指定的区间内均匀分布初始化

- **Xavier初始化(Gloart初始化)**
目的：根据输入输出神经元数量调整权重的初始范围，确保每一层输入输出的方差相近

Xavier正态分布初始化：均值为0，标准差为$\sqrt{\frac{2}{{n_{in}}+{n_{out}}}}$的正态分布
Xavier均匀分布初始化：区间$\left [{−\sqrt{\frac{6}{{n_{in}}+{n_{out}}}},\sqrt{\frac{6}{{n_{in}}+{n_{out}}}}}\right ]$均匀分布
适合于**Sigmoid和Tanh**等激活函数(饱和型激活函数)

- **He初始化(Kaiming初始化)**
He初始化根据输入神经元数量调整权重初始范围

He正态分布初始化：均值为0，标准差为$\sqrt{\frac{2}{{n_{in}}}}$的正态分布
He均匀分布初始化：区间$\left [{−\sqrt{\frac{6}{{n_{in}}}},\sqrt{\frac{6}{{n_{in}}}}}\right ]$ 的均匀分布
适合于**ReLU及其变体**

#### 4 正则化 -- 解决过拟合问题

常见正则化方法有Batch Nomalization、权值衰减、Dropout、早停法等

##### 4.1 Batch Nomalization标准化

BN放在**全连接层/卷积层**之后**激活函数**之前
![[Pasted image 20260521204241.png]]
为什么BN能够抑制过拟合：因为BN通过均值方差归一化数据，而**每一个Batch的该统计量不同**，因此给网络引入了**微小扰动**，从而起到轻微正则化作用
- 加快收敛速度
- 抑制过拟合
- 不依赖初值

##### 4.2 权值衰减

**L1和L2正则化**，都是对**损失函数**进行"惩罚"，即在损失后面加上一个数。

![[Pasted image 20260521205652.png]]
其公式为：
$$L\text{=L+ }\frac{1}{2}λ{{Penalty}}$$
λ为正则化超参数

##### 4.3 Dropout 随机失活

随机关掉部分神经元，然后对网络进行正常的训练
**可以增加网络的鲁棒性，同时防止过拟合** 一般放在**激活函数**之后，**全连接/卷积层**之前





