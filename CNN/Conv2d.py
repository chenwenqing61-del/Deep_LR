import torch
import matplotlib.pyplot as plt

img = plt.imread("data/duck.png")
print("图像数据的大小：",img.shape)

input = torch.tensor(img).permute(2,0,1).float()
print("输入尺寸大小:",input.shape)

conv = torch.nn.Conv2d(in_channels=4,out_channels=4,kernel_size=9,stride=3,padding=0,bias=False)

output=conv(input)

print("输出数据图像尺寸：",output.shape)
#转换为图片
fig,ax= plt.subplots(1,4,figsize=(10,5))
for i in range(4):
    feature_img=output[i].detach().cpu().numpy()
    ax[i].imshow(feature_img)

plt.axis("off")
plt.show()
