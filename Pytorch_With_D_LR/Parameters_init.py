import torch.nn as nn

linear = nn.Linear(5,2)
# nn.init.zeros_(linear.weight)
# print(linear.weight)

# nn.init.ones_(linear.weight)
# print(linear.weight)

# nn.init.constant_(linear.weight,10)
# print(linear.weight)

# nn.init.eye_(linear.weight)
# print(linear.weight)

# nn.init.normal_(linear.weight,mean=0.0,std=1.0)
# nn.init.uniform(linear.weight,a=0,b=10)
# print(linear.weight)


# import torch
# import torch.nn as nn
# dropout = nn.Dropout(p=0.5)
# x = torch.randint(1,10,(10,),dtype=torch.float32)
# print("Dropout前：",x)
# print("Dropout后：",dropout(x))