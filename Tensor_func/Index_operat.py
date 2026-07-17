import torch
#简单索引，直接通过tensor[,,]进行拿取数据
tensor1 = torch.randint(1,9,(3,4,5))
print(tensor1)

print(tensor1[1,:,0:3])

#范围索引，通过加上:，从而赋予从a到b-1的作用
print(tensor1[1:])
print(tensor1[-1:,1:4,0:3])

#列表索引，通过给索引的位置传入一个列表来指定某一个位置的元素
print(tensor1[[1,2],[2,3]])

print(tensor1[[[0],[1]],[1,2]])#第0维第0和第一维1，2 和第0维第1和第一维1 2

#布尔索引
print()
print(tensor1)
print(tensor1[:,:,0]>5) #表示将第2维第0大于五的 返回的是一个(3,4)矩阵
print(tensor1[tensor1[:,:,0]>5]) #返回的是值为真对应的位置，而这个位置就是对应的索引，例如[1,2]为真，则取tensor1[1,2]

tensor2 = tensor1.permute(0,2,1)
print(tensor2)
