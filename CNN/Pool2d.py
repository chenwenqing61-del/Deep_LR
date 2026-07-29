import torch
import matplotlib.pyplot as plt

img = plt.imread("data/duck.png")
print("图像数据形状：",img.shape)

input = torch.tensor(img).permute(2,0,1)
print("输入数据形状：",input.shape)

conv = torch.nn.Conv2d(in_channels=4,out_channels=4,stride=3,padding=0,kernel_size=9,bias=False)

output1 = conv(input)
print("卷积后数据形状：",output1.shape)

pool = torch.nn.MaxPool2d(kernel_size=6,stride=6,padding=1)

output2 = pool(input)

fig,axes = plt.subplots(2,4,figsize=(16,4))

for i in range(2):
    for j in range(4):
        if i==0:
            feature_img=output1[j].detach().numpy()
            axes[i,j].imshow(feature_img)
            axes[i,j].set_title("{j}通道")
        if i==1:
            feature_img=output2[j].detach().numpy()
            axes[i,j].imshow(feature_img)
            axes[i,j].set_title("{j}通道")

plt.legend()
plt.show()


