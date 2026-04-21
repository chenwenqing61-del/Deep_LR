import numpy as np
class Affine:
    def __init__(self,w,b):
        self.w =w
        self.b = b
        self.x =None
        self.dw = None
        self.db = None
        self.x_origin_shgape = None
    def forword(self,x):
        self.x = x
        self.x_origin_shape = x.shape
        out = np.dot(self.x,self.w)+self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        dx = dx.reshape(*self.x_origin_shape)
        return dx

