from numpy import *
input = array([[1,0],[1,1],[0,0],[0,1]])
target = array([[1],[0],[0],[1]])

from python.mlp import *
net = mlp(input, target, 3, outtype='logistic')
p = net.mlptrain(input, target, 0.25, 300, pl=True)
from pylab import *
plot(p[:,0], p[:,1], '.')


output = net.produce(input)
print(output)

show()


