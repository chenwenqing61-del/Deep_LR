import torch
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X,optimizer,n_iters):
    X_arr = X.detach().numpy().copy()
    for epoch in range(n_iters):
        y = X**2 @ w
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        X_arr = np.vstack([X_arr,X.detach().numpy()])
    return X_arr

X = torch.tensor([-7,-2],dtype=torch.float32,requires_grad=True)
w = torch.tensor([0.05,1],requires_grad=True)
lr = 1e-2
n_iters=500
X_clone = X.clone().detach().requires_grad_(True)
X_arr1 = gradient_descent(X_clone,torch.optim.SGD([X_clone],lr=lr),n_iters=n_iters)
plt.plot(X_arr1[:,0],X_arr1[:,1],"r")

X_clone = X.clone().detach().requires_grad_(True)
X_arr2 = gradient_descent(X_clone,torch.optim.SGD([X_clone],lr=lr,momentum=0.9),n_iters=n_iters)
plt.plot(X_arr2[:,0],X_arr2[:,1],"b")

x1_grid,x2_grid = np.meshgrid(np.linspace(-7,7,100),np.linspace(-2,2,100))
y_grid = w.detach().numpy()[0] * x1_grid**2 + w.detach().numpy()[1]*x2_grid**2

plt.contour(x1_grid,x2_grid,y_grid,levels=30,colors="gray")
plt.legend(["SGD","Momentum"])
plt.show()

