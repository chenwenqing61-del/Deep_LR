import numpy as np

class SoftmaxWithEntropy:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self,x,t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y-self.t)/batch_size
        else:
            dx = np.copy(self.y)
            dx[np.arange(batch_size),self.t] -=1
            dx = dx/batch_size
        return dx