from numpy import *

class linreg:

    def train(self, inputs, targets):
        inputs = concatenate((-ones((shape(inputs)[0],1)),inputs),axis=1)
        self.beta = dot(dot(linalg.inv(dot(transpose(inputs),inputs)),transpose(inputs)),targets)
        #print(dot(inputs,self.beta))

    def getError(self, inputs, targets):
        inputs = concatenate((-ones((shape(inputs)[0], 1)), inputs), axis=1)
        testout = dot(inputs, self.beta)
        error = sum((testout - targets)**2)
        print(concatenate((dot(inputs,self.beta), targets), axis=1))
        return error


auto = loadtxt('auto-mpg.data', comments='"')
# trainin = auto[::2,1:]
# trainout = auto[::2,0:1]
# testin = auto[1::2,1:]
# testout = auto[1::2,0:1]

p = linreg()
# p.train(trainin, trainout)
# p.getError(testin, testout)
p.train(auto[:,1:], auto[:,0:1])
p.getError(auto[:,1:], auto[:,0:1])

