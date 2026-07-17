import torch

tensor1 = torch.randint(1,9,(3,4,5,1,1))
# print(tensor1)
# print(tensor1.transpose(1,2))
# print(tensor1)
# print(tensor1.permute(1,2,0))

# print(tensor1)
# print(tensor1.reshape(6,10))
# print(tensor1.reshape(3,-1))

# print(tensor1.is_contiguous())
# print(tensor1.view(-1,10))
# tensor1 = tensor1.view(-1,10)
# tensor1 = tensor1.T
# print(tensor1.is_contiguous())
# tensor1 = tensor1.contiguous()
# print(tensor1.is_contiguous())
print(tensor1.size())
print(tensor1.squeeze_(dim=-1),tensor1.size())