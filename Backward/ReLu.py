class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        y = x.copy()
        y[self.mask] = 0
        return y
    
    def backward(self,dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
