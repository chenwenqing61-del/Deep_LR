import numpy as np
from gradient import _numerical_gradient

def gradient_decent(f,ini_x,lr=0.01,step_num=100):
    x = ini_x
    x_history =[]
    for i in range(step_num):
        x_history.append(x)
        grad = _numerical_gradient(f,x)
        x -=lr*grad
    return x,np.array(x_history)
