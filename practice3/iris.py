from numpy import *

def preprocessIris(infile,outfile):

    stext1 = 'Iris-setosa'
    stext2 = 'Iris-versicolor'
    stext3 = 'Iris-virginica'
    rtext1 = '0'
    rtext2 = '1'
    rtext3 = '2'

    fid = open(infile,"r")
    oid = open(outfile,"w")

    for s in fid:
        if s.find(stext1)>-1:
            oid.write(s.replace(stext1, rtext1))
        elif s.find(stext2)>-1:
            oid.write(s.replace(stext2, rtext2))
        elif s.find(stext3)>-1:
            oid.write(s.replace(stext3, rtext3))
    fid.close()
    oid.close()

preprocessIris('iris.data','iris_proc.data')

iris = loadtxt('iris_proc.data', delimiter = ',')

iris[:,:4] = iris[:,:4] - iris[:,:4].mean(axis=0)
imax = concatenate((iris.max(axis=0)*ones((1,5)),abs(iris.min(axis=0)*ones((1,5)))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]
print(iris[0:5,:])

target = zeros((shape(iris)[0],3))
indices = where(iris[:,4]==0)
print(shape(indices))
target[indices,0] = 1
indices = where(iris[:,4]==1)
target[indices,1] = 1
indices = where(iris[:,4]==2)
target[indices,2] = 1

order = list(range(shape(iris)[0]))
random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
testi = iris[3::4,0:4]
testt = target[3::4]
from python.mlp import *
net = mlp(train,traint,5,outtype='softmax')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(testi,testt)

from pylab import *
