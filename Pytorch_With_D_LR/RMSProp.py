import torch
import numpy as np
import matplotlib.pyplot as plt

def rmsprop(X,lr,alpha,n_iters):
    X_arr = X.detach().numpy().copy()
    H = torch.zeros_like(X)
    for epoch in range(n_iters):
        grad = 2*X * w.T
        grad.squeeze_()
        H = alpha*H +(1-alpha)*grad
        X.data -= lr/(torch.sqrt(H)+1e-7) *grad
        X_arr = np.vstack([X_arr,X.detach().numpy()])
    return X_arr

def gradient_descent(X,optimizer,n_iters):
    X_arr = X.detach().numpy().copy()
    for epoch in range(n_iters):
        y = X**2 @ w
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        X_arr = np.vstack([X_arr,X.detach().numpy()])
    return X_arr

X = torch.tensor([-7,2],dtype=torch.float32,requires_grad=True)
w = torch.tensor([[0.05],[1.0]],requires_grad=True)
lr=1e-1
n_iters=1000

#普通梯度下降
X_clone = X.clone().detach().requires_grad_(True)
X_arr1 = gradient_descent(X_clone,torch.optim.SGD([X_clone],lr=lr),n_iters=n_iters)
plt.plot(X_arr1[:,0],X_arr1[:,1],'r')

#RMSProp
X_clone = X.clone().detach().requires_grad_()
X_arr2 = gradient_descent(X_clone,torch.optim.RMSprop([X_clone],lr=lr,alpha=0.99),n_iters=n_iters)
plt.plot(X_arr2[:,0],X_arr2[:,1],'b')

#手动RMSProp
X_clone = X.clone().detach().requires_grad_()
X_arr3 = rmsprop(X_clone,lr,0.99,n_iters)
plt.plot(X_arr3[:,0],X_arr3[:,1],c="orange",linestyle="--",linewidth=3)

x1_grid,x2_grid = np.meshgrid(np.linspace(-7,7,100),np.linspace(-2,2,100))
y_grid = w.detach().numpy()[0,0]*x1_grid**2+w.detach().numpy()[1,0]*x2_grid**2
plt.contour(x1_grid,x2_grid,y_grid,levels=30,colors="grey")
plt.legend(["SGD","RMSProp","Manual RMSProp"])
plt.show()