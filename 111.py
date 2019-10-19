import numpy as np
import random as rd
from pylab import *
x=np.linspace(-100,100,20)
y=[i*i+rd.uniform(-i*i/10,i*i/10) for i in x]
class net:
    def __init__(self):
        self.w1=rd.random()
        self.w2=rd.random()
        self.w3=rd.random()
    def out(self,x):
        return self.w1*x*x+self.w2*x+self.w3
    def grad(self,inp,outp):
        g1=0
        g2=0
        g3=0
        for i in range(len(inp)):
           g1+=(outp[i]-(self.out(inp[i])))*(-inp[i]**2)
           g2+=(outp[i]-(self.out(inp[i])))*(-inp[i])
           g3+=(outp[i]-(self.out(inp[i])))*-1
        return g1,g2,g3
    def train(self,inp,outp,eta):
        loss=[]
        times=[]
        g1s=0
        g2s=0
        g3s=0
        for i in range(1000):
            s=0
            for j in range(len(inp)):
                s+=(outp[j]-self.out(inp[j]))**2
            loss.append(s)
            times.append(i)
            g1,g2,g3=self.grad(inp,outp)
            g1s+=g1**2
            g2s+=g2**2
            g3s+=g3**2
            self.w1-=eta*g1/sqrt(g1s)
            self.w2-=eta*g2/sqrt(g2s)
            self.w3-=eta*g3/sqrt(g3s)
        subplot(211)
        plot(times,loss)
work=net()
work.train(x,y,0.1)
print(work.w1,work.w2,work.w3)
z=[work.out(i) for i in x]
subplot(212)
plot(x,y,x,z)
show()
