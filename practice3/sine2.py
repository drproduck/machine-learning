from pylab import *

from python.mlp import *
from numpy import *

x = ones((1,100))*linspace(0,1,100)
random.shuffle(x)
t = sin(2*pi*x)+ cos(4*pi*x) + random.randn(100)*0.2
x = transpose(x)
t = transpose(t)

mean = x.mean(axis=0)
var = x.var(axis=0)
x = (x - mean)/var

meant = t.mean(axis=0)
vart = t.var(axis=0)
t = (t - meant)/vart

plot(x,t,'.')
figure()

train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

net = mlp(train, traintarget, 3, outtype='linear')
p = net.mlptrain(train, traintarget, eta=0.25, niterations=1000, pl=True)

print(p)
plot(p[:,0], p[:,1], '.')
show()