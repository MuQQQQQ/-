import numpy as np
import random as rd
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
def errors(a,b,x,y):
    error=sum((a*x+b-y)**2)
    #error=sum(np.exp(-(a*x+b)*y))
    return error
x=np.linspace(1,100,300)
y=3*x+5
hy=[i+rd.uniform(-i/2,i/2) for i in y]
#show()
a=np.linspace(2.5,3.5,100)
b=np.linspace(4,6,100)
A,B=np.meshgrid(a,b)
fig=figure()
ax=Axes3D(fig)
Z=np.zeros(A.shape)
for i in range(len(a)):
    for j in range(len(b)):
        Z[i][j]=errors(a[i],b[j],x,hy)
ax.plot_surface(A,B,Z,cmap='rainbow')
ax.contour(A,B,Z,zdir='z', offset=-3,cmap="rainbow")
show()
