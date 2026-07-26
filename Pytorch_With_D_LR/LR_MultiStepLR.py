import torch
import numpy as np
import matplotlib.pyplot as plt

X = torch.tensor([-7,-2],dtype=torch.float32,requires_grad=True)
w = torch.tensor([[0.05],[1.0]],requires_grad=True)
lr = 0.9
n_iters= 400

optimizer = torch.optim.SGD([X],lr=lr)
scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [10,50,200],gamma=0.7)
X_arr = X.detach().numpy().copy()
lr_list = []
for epoch in range(n_iters):
    y = X**2 @ w
    y.backward()
    optimizer.step()
    optimizer.zero_grad()
    X_arr = np.vstack([X_arr,X.detach().numpy()])
    lr_list.append(optimizer.param_groups[0]["lr"])
    scheduler_lr.step()

plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
fig,ax = plt.subplots(1,2,figsize=(12,4))

x1_grid,x2_grid = np.meshgrid(np.linspace(-7,7,100),np.linspace(-2,2,100))
y_grid = w.detach().numpy()[0,0]* x1_grid**2 + w.detach().numpy()[1,0]* x2_grid**2

ax[0].contour(x1_grid,x2_grid,y_grid,levels=30,colors="grey")
ax[0].plot(X_arr[:,0],X_arr[:,1],"r")
ax[0].set_title("梯度下降过程")

ax[1].plot(lr_list,"k")
ax[1].set_title("学习率衰减")
plt.show()
