import torch
import numpy as np
#使用torch.tensor()直接创建张量，可直接带入数据/用数组/向量进行初始化
'''
tensor1 = torch.tensor(10)
print(tensor1)

tensor2 = torch.tensor([10,9,8])
print(tensor2)

tensor2 = torch.tensor(np.array([[1,2,3],[4,5,6]]))
print(tensor2,tensor2.shape)
'''


#创建指定形状的张量 torch.Tensor(size)/(([])指定内容)
'''
tensor1 = torch.Tensor(3,2,4)
print(tensor1,tensor1.shape,tensor1.dtype)
tensor2 = torch.Tensor([[1,2,3],[4,5,6]])
print(tensor2,tensor2.shape,tensor2.dtype)
'''

#创建指定类型的张量
'''
tensor1 = torch.IntTensor(2,3,4)
print(tensor1,tensor1.shape,tensor1.dtype)
'''

#arange(begin,end,step)从begin to end按照步长进行创建张量
'''
tensor1 = torch.arange(1,10,2)
print(tensor1)
tensor2 = torch.arange(6)
print(tensor2)
'''
#linspace(begin,end,steps)从begin to end 创建steps个点的张量
'''
tensor1 = torch.linspace(1,10,5)
print(tensor1)
'''
#logspace(begin,end,steps,base)从begin to end创建steps个元素，并将元素作为base的指数
'''
tensor1 = torch.logspace(1,10,5,base=2)
print(tensor1)
'''

#按数值填充张量 全1 全0 全values   eye(n,m)创建单位矩阵:若n!=m则有几列/几行全为0
'''
tensor1 = torch.ones(2,3,4)
tensor2 = torch.zeros(2,1,2)
tensor3 = torch.full((2,3),3)
print(tensor1,tensor1.shape)
print(tensor2,tensor2.shape)
print(tensor3,tensor3.shape)

print(torch.zeros_like(tensor1))
print(torch.ones_like(tensor2))
print(torch.full_like(tensor3,5))

tensor4 = torch.eye(3,4)
print(tensor4,tensor4.shape)
'''

#随机张量的创建 
'''
rand创建0~1的均匀分布
randn创建标准正态分布
randint(low,high,size)创建从low~high的均匀分布整数
normal(mean,std,size)创建服从均值为mean,方差为std的正态分布
rand_like() randint_like() randn_like() 等表示创建与传入张量形状相同的随机张量
'''
'''
tensor1 = torch.rand(2,3,4)
tensor2 = torch.randn(2,3)
tensor3 = torch.randint(1,10,(2,3))
tensor4 = torch.normal(2,1,(3,2))
print(tensor1,tensor1.shape)
print(tensor2,tensor2.shape)
print(tensor3,tensor3.shape)
print(tensor4,tensor4.shape)
'''

#随机种子的设置与获取 torch.random.initial_seed()获取当前随机种子，torch.manual_seed(seed)设置随机种子
'''
print(torch.random.initial_seed())
torch.manual_seed(52)
print(torch.random.initial_seed())
'''

