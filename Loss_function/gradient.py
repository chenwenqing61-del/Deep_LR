import numpy as np
def _numerical_gradient(f, x):
    #计算高维度函数的数值梯度-- 中心差分法 
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size()):
        x_val = x[i]
        x[i] = float(x_val)+h
        f_plus = f(x[i])
        x[i] = float(x_val)-h
        f_minus = f(x[i])
        grad[i] = (f_plus+f_minus)/(2*h)
        x[i] = x_val  #还原x的值
    return grad

