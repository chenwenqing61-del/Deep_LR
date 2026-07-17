import torch
import numpy as np
'''
对张量进行运算时，只有当运算后面有_时，才会对原来的张量进行修改
否则，不会对原来的张量进行修改，而是返回一个新的张量
add()/+,sub()/-,mul()/*,div()使用/进行替换
取反neg()/-
求幂**,pow(),pow_()
开次方sqrt(),sqrt_()
求e对数log(),log_()
求e为底的指数exp(),exp_()
求绝对值abs(),abs_()
'''

# tensor1 = torch.tensor([1,2,3])
# print(tensor1.sub(1))
# print(tensor1)


'''
哈达玛积  --  基本乘法,对应位置相乘
要求两个矩阵size必须完全相同
mul()/*

矩阵乘法  --  严格的(3,2)×(3,2)=(2,2)
mm()严格二维矩阵相乘
matual()和@支持多维矩阵的乘法，对矩阵的最后两个维度进行矩阵相乘
'''

# tensor1 = torch.tensor([[1,2,3],[4,5,6]])
# tensor2 = torch.tensor([[1,2],[3,4],[5,6]])
# print(tensor1@tensor2)


'''
内存节省方法，在进行矩阵乘法时
若使用X=X@Y会新开一个地址用于存储计算结果
X[:]=X@Y将不会新开一个地址而是将计算结果直接
必须要求进行X@Y计算的size与X相同
'''
# tensor1 = torch.tensor([1,2,3])
# tensor2 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(id(tensor1))
# tensor1[:]=tensor1@tensor2
# print(id(tensor1))

#张量的统计学函数
'''
tensor1.sum()这样使用
sum()求和
mean()均值
max()/min()求最大值/最小值及其索引
argmax()/argmin()求最大值/最小值索引
std()标准差
unique()去重
sort()排序
'''
tensor1 = torch.tensor(np.array([7,2,2,3,3,54]))
print(tensor1.sum())
print(tensor1.max())
print(tensor1.argmax())
print(tensor1.unique())


