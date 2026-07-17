import torch
tensor1 = torch.randint(1,9,(3,4,5))
tensor2 = torch.randint(1,9,(3,1,5))
# print(torch.cat([tensor1,tensor2],dim=1))
tensor3 = torch.randint(1,9,(3,4,5))
tensor4 = torch.stack([tensor1,tensor3],dim=1)
print(tensor4,tensor4.shape)