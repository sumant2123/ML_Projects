import numpy as np
from scipy.special import softmax
from scipy.special import log_softmax

data = np.genfromtxt("HamiltonCases.csv", delimiter=',')
X = data[:, 0:4]
y = data[:, 4]
z0 = (y < X.min(axis=1)).astype(int)
z2 = (y > X.max(axis=1)).astype(int)
z1 = ((z0 == 0) & (z2 == 0)).astype(int)
y = np.concatenate([z0[:, np.newaxis], z1[:, np.newaxis], z2[:, np.newaxis]], axis=1)
X = np.concatenate([X, np.ones(len(X))[:, np.newaxis]], axis=1)
m = 200
Xtrain = X[0:m, :]
ytrain = y[0:m, :]
target = np.argmax(ytrain, axis=1)
theta = np.random.randn(5, 3)
o = np.matmul(Xtrain, theta)
predict = np.argmax(o, axis=1)
before = np.sum(predict == target)
eta = 0.001
n_iterations = 1000
for i in range(n_iterations):
    # o= x.Θ
    o = np.matmul(Xtrain, theta)

    # y~ =softmax(o)
    p = softmax(o, axis=1)

    # NLL =-log(y~) -> -log(softmax(o,axis=1)) -> log_softmax(o)
    NLL = -log_softmax(o, axis=1)

    # L= y*(-log(y^))
    #	=- y*(NLL)
    L = np.tensordot(NLL, ytrain, axes=[[0, 1], [0, 1]]) / m
    print(L)

    # 
# (y^ - y).x = Xtrain.(p-ytrain)

gradient = np.matmul(Xtrain.T, (p - ytrain)) / m
"""

theta = theta -eta * gradient
"""
theta = theta - eta * gradient
print(theta)

o = np.matmul(Xtrain, theta)
predict = np.argmax(o, axis=1)
print(before, np.sum(predict == target))

Xtest = X[m:, :]
ytest = y[m:, :]
o = np.matmul(Xtest, theta)
predict = np.argmax(o, axis=1)

act = np.argmax(ytest, axis=1)
print(np.sum(predict == act), np.sum(predict != act))
