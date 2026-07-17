import torch
import numpy as np
#修改元素类型 tensor1.type(torch.int32)等/ 直接使用tensor1.short()进行转换
'''
tensor1 = torch.tensor([1,2,3])
print(tensor1,tensor1.dtype)
print(tensor1.type(torch.int32))
print(tensor1.double())
'''

#tensor与nparray之间相互的转换
'''
转换涉及共享内存的概念，即当直接进行转换时，tensor与nparray是使用的同一个内存地址
对其中一个修改另一个也会被修改，可以使用copy()方法进行不共享内存
!!!! copy方法一定是numpy的copy()方法，而不是tensor的copy()方法
tensor转np，使用tensor1.numpy()进行转换，此时有共享内存；使用tensor1.numpy().copy()不共享内存
np转tensor，使用torch.from_numpy(np1)共享内存；使用torch.from_numpy(np1.copy())不共享内存
'''
'''
tensor1 = torch.tensor([1,2,3])
np1 = np.array([2,3,4])
tensor2 = torch.from_numpy(np1)
tensor2[2] = 5
print(tensor2)
print(np1)
tensor2 = torch.from_numpy(np1.copy())
tensor2[2] = 2
print(tensor2)
print(np1)

np2 = tensor1.numpy()
np2[1] = 5
print(np2)
print(tensor1)
np2 = tensor1.numpy().copy()
np2[1] = 2
print(np2)
print(tensor1)
'''

#item()方法将单元素张量转换为数值类型
tensor1 = torch.tensor(1)
print(tensor1.item())
