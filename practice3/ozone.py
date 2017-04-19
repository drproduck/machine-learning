import pylab as pl
import numpy as np

PNoz = np.loadtxt('PNoz.data')
pl.ion()
pl.plot(np.arange(np.shape(PNoz)[0]), PNoz[:, 2], '.')
pl.xlabel('Time (Days)')
pl.ylabel('Ozone (Dobson units)')

# Normalise data
PNoz[:, 2] = PNoz[:, 2] - PNoz[:, 2].mean()
PNoz[:, 2] = PNoz[:, 2] / PNoz[:, 2].max()

# Assemble input vectors
t = 2
k = 3

lastPoint = np.shape(PNoz)[0] - t * (k + 1)
inputs = np.zeros((lastPoint, k))
targets = np.zeros((lastPoint, 1))
for i in range(lastPoint):
    inputs[i, :] = PNoz[i:i + t * k:t, 2]
    targets[i] = PNoz[i + t * (k + 1), 2]

testi = inputs[-400:, :]
testtargets = targets[-400:]
train = inputs[:-400:2, :]
traintargets = targets[:-400:2]
valid = inputs[1:-400:2, :]
validtargets = targets[1:-400:2]

# Randomly order the data
change = list(range(np.shape(inputs)[0]))
np.random.shuffle(change)
inputs = inputs[change, :]
targets = targets[change, :]

# Train the network
from python.mlp import *

net = mlp(train, traintargets, 3, outtype='linear')
net.earlystopping(train, traintargets, valid, validtargets, 0.25)

testi = np.concatenate((testi, -np.ones((np.shape(testi)[0], 1))), axis=1)
testout = net.mlpfwd(testi)

pl.figure()
pl.plot(np.arange(np.shape(testi)[0]), testout, '.')
pl.plot(np.arange(np.shape(testi)[0]), testtargets, 'x')
pl.legend(('Predictions', 'Targets'))
print(0.5 * np.sum((testtargets - testout) ** 2))

pl.show(block=True)