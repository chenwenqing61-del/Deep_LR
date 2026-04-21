import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forword(self,x):
        self.out = 1/(1+np.exp(-x)) #sigmoid函数
        return self.out

    def backward(self,dout):
        dx = dout*(1-self.out)*self.out
        return dx
    
